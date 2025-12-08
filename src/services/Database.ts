/**
 * PGlite Database Service with pgvector for semantic memories
 *
 * Uses PGlite (WASM Postgres) with pgvector for vector similarity search.
 * Simplified schema for storing memories with embeddings.
 */

import { Effect, Context, Layer, Data } from "effect";
import { PGlite } from "@electric-sql/pglite";
import { vector } from "@electric-sql/pglite/vector";
import { mkdirSync, existsSync } from "fs";
import { dirname } from "path";

// ============================================================================
// Types (inline until types.ts is ready)
// ============================================================================

/** Embedding dimension for mxbai-embed-large */
const EMBEDDING_DIM = 1024;

/** Memory data structure */
export interface Memory {
  readonly id: string;
  readonly content: string;
  readonly metadata: Record<string, unknown>;
  readonly collection: string;
  readonly createdAt: Date;
}

/** Search result with similarity score */
export interface SearchResult {
  readonly memory: Memory;
  readonly score: number;
  readonly matchType: "vector" | "fts";
}

/** Search options for queries */
export interface SearchOptions {
  readonly limit?: number;
  readonly threshold?: number;
  readonly collection?: string;
}

/** Database error with reason */
export class DatabaseError extends Data.TaggedError("DatabaseError")<{
  readonly reason: string;
}> {}

/** Memory configuration */
export interface MemoryConfig {
  readonly dbPath: string;
}

// ============================================================================
// Service Definition
// ============================================================================

export class Database extends Context.Tag("Database")<
  Database,
  {
    /** Store a memory with its embedding */
    readonly store: (
      memory: Memory,
      embedding: number[],
    ) => Effect.Effect<void, DatabaseError>;

    /** Vector similarity search */
    readonly search: (
      embedding: number[],
      options?: SearchOptions,
    ) => Effect.Effect<SearchResult[], DatabaseError>;

    /** Full-text search */
    readonly ftsSearch: (
      query: string,
      options?: SearchOptions,
    ) => Effect.Effect<SearchResult[], DatabaseError>;

    /** List memories, optionally filtered by collection */
    readonly list: (
      collection?: string,
    ) => Effect.Effect<Memory[], DatabaseError>;

    /** Get a single memory by ID */
    readonly get: (id: string) => Effect.Effect<Memory | null, DatabaseError>;

    /** Delete a memory by ID */
    readonly delete: (id: string) => Effect.Effect<void, DatabaseError>;

    /** Get database statistics */
    readonly getStats: () => Effect.Effect<
      { memories: number; embeddings: number },
      DatabaseError
    >;
  }
>() {}

// ============================================================================
// Implementation
// ============================================================================

/**
 * Create a Database layer with the given configuration
 */
export const makeDatabaseLive = (config: MemoryConfig) =>
  Layer.scoped(
    Database,
    Effect.gen(function* () {
      // Ensure directory exists
      const dbDir = dirname(config.dbPath);
      if (!existsSync(dbDir)) {
        mkdirSync(dbDir, { recursive: true });
      }

      // PGlite stores data in a directory, not a single file
      const pgDataDir = config.dbPath.replace(".db", "");

      // Initialize PGlite with pgvector extension
      const db = yield* Effect.tryPromise({
        try: () =>
          PGlite.create({
            dataDir: pgDataDir,
            extensions: { vector },
          }),
        catch: (e) =>
          new DatabaseError({ reason: `Failed to init PGlite: ${e}` }),
      });

      // Initialize schema
      yield* Effect.tryPromise({
        try: async () => {
          // Enable pgvector
          await db.exec("CREATE EXTENSION IF NOT EXISTS vector;");

          // Memories table
          await db.exec(`
            CREATE TABLE IF NOT EXISTS memories (
              id TEXT PRIMARY KEY,
              content TEXT NOT NULL,
              metadata JSONB DEFAULT '{}',
              collection TEXT DEFAULT 'default',
              created_at TIMESTAMPTZ DEFAULT NOW()
            )
          `);

          // Memory embeddings table with vector column
          await db.exec(`
            CREATE TABLE IF NOT EXISTS memory_embeddings (
              memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
              embedding vector(${EMBEDDING_DIM}) NOT NULL
            )
          `);

          // Create HNSW index for fast approximate nearest neighbor search
          await db.exec(`
            CREATE INDEX IF NOT EXISTS memory_embeddings_hnsw_idx 
            ON memory_embeddings 
            USING hnsw (embedding vector_cosine_ops)
          `);

          // Full-text search index on content
          await db.exec(`
            CREATE INDEX IF NOT EXISTS memories_content_idx 
            ON memories 
            USING gin (to_tsvector('english', content))
          `);

          // Index for collection filtering
          await db.exec(
            `CREATE INDEX IF NOT EXISTS idx_memories_collection ON memories(collection)`,
          );
        },
        catch: (e) => new DatabaseError({ reason: `Schema init failed: ${e}` }),
      });

      // Cleanup on scope close
      yield* Effect.addFinalizer(() =>
        Effect.promise(async () => {
          await db.close();
        }),
      );

      // Helper to parse memory row
      const parseMemoryRow = (row: any): Memory => ({
        id: row.id,
        content: row.content,
        metadata: row.metadata ?? {},
        collection: row.collection ?? "default",
        createdAt: new Date(row.created_at),
      });

      return {
        store: (memory, embedding) =>
          Effect.tryPromise({
            try: async () => {
              await db.exec("BEGIN");
              try {
                // Insert or update memory
                await db.query(
                  `INSERT INTO memories (id, content, metadata, collection, created_at)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (id) DO UPDATE SET
                     content = EXCLUDED.content,
                     metadata = EXCLUDED.metadata,
                     collection = EXCLUDED.collection`,
                  [
                    memory.id,
                    memory.content,
                    JSON.stringify(memory.metadata),
                    memory.collection,
                    memory.createdAt.toISOString(),
                  ],
                );

                // Insert or update embedding
                const vectorStr = `[${embedding.join(",")}]`;
                await db.query(
                  `INSERT INTO memory_embeddings (memory_id, embedding)
                   VALUES ($1, $2::vector)
                   ON CONFLICT (memory_id) DO UPDATE SET
                     embedding = EXCLUDED.embedding`,
                  [memory.id, vectorStr],
                );

                await db.exec("COMMIT");
              } catch (e) {
                await db.exec("ROLLBACK");
                throw e;
              }
            },
            catch: (e) => new DatabaseError({ reason: String(e) }),
          }),

        search: (queryEmbedding, options = {}) =>
          Effect.tryPromise({
            try: async () => {
              const { limit = 10, threshold = 0.3, collection } = options;

              // Format query vector
              const vectorStr = `[${queryEmbedding.join(",")}]`;

              let query = `
                SELECT 
                  m.id,
                  m.content,
                  m.metadata,
                  m.collection,
                  m.created_at,
                  1 - (e.embedding <=> $1::vector) as score
                FROM memory_embeddings e
                JOIN memories m ON m.id = e.memory_id
              `;

              const params: any[] = [vectorStr];
              let paramIdx = 2;

              // Collection filter
              if (collection) {
                query += ` WHERE m.collection = $${paramIdx}`;
                params.push(collection);
                paramIdx++;
              }

              // Threshold filter
              if (collection) {
                query += ` AND 1 - (e.embedding <=> $1::vector) >= $${paramIdx}`;
              } else {
                query += ` WHERE 1 - (e.embedding <=> $1::vector) >= $${paramIdx}`;
              }
              params.push(threshold);
              paramIdx++;

              // Order and limit
              query += ` ORDER BY e.embedding <=> $1::vector LIMIT $${paramIdx}`;
              params.push(limit);

              const result = await db.query(query, params);

              return result.rows.map((row: any) => ({
                memory: parseMemoryRow(row),
                score: row.score,
                matchType: "vector" as const,
              }));
            },
            catch: (e) => new DatabaseError({ reason: String(e) }),
          }),

        ftsSearch: (searchQuery, options = {}) =>
          Effect.tryPromise({
            try: async () => {
              const { limit = 10, collection } = options;

              let sql = `
                SELECT 
                  m.id,
                  m.content,
                  m.metadata,
                  m.collection,
                  m.created_at,
                  ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', $1)) as score
                FROM memories m
                WHERE to_tsvector('english', m.content) @@ plainto_tsquery('english', $1)
              `;

              const params: any[] = [searchQuery];
              let paramIdx = 2;

              if (collection) {
                sql += ` AND m.collection = $${paramIdx}`;
                params.push(collection);
                paramIdx++;
              }

              sql += ` ORDER BY score DESC LIMIT $${paramIdx}`;
              params.push(limit);

              const result = await db.query(sql, params);

              return result.rows.map((row: any) => ({
                memory: parseMemoryRow(row),
                score: row.score,
                matchType: "fts" as const,
              }));
            },
            catch: (e) => new DatabaseError({ reason: String(e) }),
          }),

        list: (collection) =>
          Effect.tryPromise({
            try: async () => {
              let query = "SELECT * FROM memories";
              const params: string[] = [];

              if (collection) {
                query += " WHERE collection = $1";
                params.push(collection);
              }

              query += " ORDER BY created_at DESC";

              const result = await db.query(query, params);
              return result.rows.map(parseMemoryRow);
            },
            catch: (e) => new DatabaseError({ reason: String(e) }),
          }),

        get: (id) =>
          Effect.tryPromise({
            try: async () => {
              const result = await db.query(
                "SELECT * FROM memories WHERE id = $1",
                [id],
              );
              return result.rows.length > 0
                ? parseMemoryRow(result.rows[0])
                : null;
            },
            catch: (e) => new DatabaseError({ reason: String(e) }),
          }),

        delete: (id) =>
          Effect.tryPromise({
            try: async () => {
              // Cascade handles memory_embeddings
              await db.query("DELETE FROM memories WHERE id = $1", [id]);
            },
            catch: (e) => new DatabaseError({ reason: String(e) }),
          }),

        getStats: () =>
          Effect.tryPromise({
            try: async () => {
              const memories = await db.query(
                "SELECT COUNT(*) as count FROM memories",
              );
              const embeddings = await db.query(
                "SELECT COUNT(*) as count FROM memory_embeddings",
              );

              return {
                memories: Number((memories.rows[0] as { count: number }).count),
                embeddings: Number(
                  (embeddings.rows[0] as { count: number }).count,
                ),
              };
            },
            catch: (e) => new DatabaseError({ reason: String(e) }),
          }),
      };
    }),
  );

/**
 * Default Database layer using environment config
 * Expects MEMORY_DB_PATH env var or defaults to ~/.semantic-memory/memory.db
 */
export const DatabaseLive = makeDatabaseLive({
  dbPath:
    process.env.MEMORY_DB_PATH ??
    `${process.env.HOME}/.semantic-memory/memory.db`,
});
