#!/usr/bin/env bun
/**
 * Semantic Memory CLI
 *
 * Store and retrieve memories with semantic search.
 */

import { Effect, Console, Layer } from "effect";
import { randomUUID } from "crypto";
import { existsSync } from "fs";
import { Database, makeDatabaseLive } from "./services/Database.js";
import { Ollama, makeOllamaLive } from "./services/Ollama.js";
import { MemoryConfig } from "./types.js";
import {
  needsMigration,
  migrate,
  importMigrationDump,
  generateMigrationScript,
} from "./services/Migration.js";

// ============================================================================
// Help Text
// ============================================================================

const HELP = `
semantic-memory - Local semantic memory with vector search

Usage:
  semantic-memory <command> [options]

Commands:
  store <content>         Store a memory
    --metadata <json>     JSON metadata object
    --collection <name>   Collection name (default: "default")
    --tags <tags>         Comma-separated tags

  find <query>            Semantic search for memories
    --limit <n>           Max results (default: 10)
    --collection <name>   Filter by collection
    --fts                 Full-text search only (no embeddings)
    --expand              Return full content instead of truncated preview

  list                    List all memories
    --collection <name>   Filter by collection

  get <id>                Get a memory by ID

  delete <id>             Delete a memory by ID

  validate <id>           Validate/reinforce a memory (resets decay timer)

  stats                   Show memory statistics

  check                   Verify Ollama is running

  migrate                 Migrate database from PGlite 0.2.x to 0.3.x
    --check               Check if migration is needed
    --import <file>       Import a SQL dump file
    --generate-script     Generate a migration helper script
    --no-backup           Don't keep backup after migration

Options:
  --help, -h              Show this help
  --json                  Output as JSON

Examples:
  semantic-memory store "Meeting notes from standup" --tags "meetings,work"
  semantic-memory find "what did we discuss in standup" --limit 5
  semantic-memory list --collection work
  semantic-memory migrate --check
  semantic-memory migrate --generate-script > migrate.ts
`;

// ============================================================================
// Arg Parsing
// ============================================================================

function parseArgs(args: string[]): Record<string, string | boolean> {
  const result: Record<string, string | boolean> = {};
  let i = 0;
  while (i < args.length) {
    const arg = args[i];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const next = args[i + 1];
      if (next && !next.startsWith("--")) {
        result[key] = next;
        i += 2;
      } else {
        result[key] = true;
        i += 1;
      }
    } else {
      i += 1;
    }
  }
  return result;
}

// ============================================================================
// Main Program
// ============================================================================

const program = Effect.gen(function* () {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    yield* Console.log(HELP);
    return;
  }

  const command = args[0];
  const opts = parseArgs(args.slice(1));
  const jsonOutput = opts.json === true;

  // migrate command is handled separately before database initialization
  if (command === "migrate") {
    yield* Console.error(
      "Error: migrate command should be handled before this point",
    );
    process.exit(1);
  }

  const db = yield* Database;
  const ollama = yield* Ollama;
  const config = MemoryConfig.fromEnv();

  switch (command) {
    case "store": {
      const content = args[1];
      if (!content) {
        yield* Console.error("Error: Content required");
        process.exit(1);
      }

      const collection =
        (opts.collection as string) || config.defaultCollection;
      const tags = opts.tags
        ? (opts.tags as string).split(",").map((t) => t.trim())
        : [];

      let metadata: Record<string, unknown> = {};
      if (opts.metadata) {
        try {
          metadata = JSON.parse(opts.metadata as string);
        } catch {
          yield* Console.error("Error: Invalid JSON in --metadata");
          process.exit(1);
        }
      }

      // Add tags to metadata
      if (tags.length > 0) {
        metadata.tags = tags;
      }

      const id = randomUUID();
      const memory = {
        id,
        content,
        metadata,
        collection,
        createdAt: new Date(),
      };

      yield* Console.log("Generating embedding...");
      const embedding = yield* ollama.embed(content);

      yield* Console.log("Storing memory...");
      yield* db.store(memory, embedding);

      if (jsonOutput) {
        yield* Console.log(
          JSON.stringify({ id, content, collection, metadata }, null, 2),
        );
      } else {
        yield* Console.log(`✓ Stored memory`);
        yield* Console.log(`  ID: ${id}`);
        yield* Console.log(`  Collection: ${collection}`);
        if (tags.length) yield* Console.log(`  Tags: ${tags.join(", ")}`);
      }
      break;
    }

    case "find": {
      const query = args[1];
      if (!query) {
        yield* Console.error("Error: Query required");
        process.exit(1);
      }

      const limit = opts.limit ? parseInt(opts.limit as string, 10) : 10;
      const collection = opts.collection as string | undefined;
      const ftsOnly = opts.fts === true;
      const expand = opts.expand === true;

      // Read decay half-life from env
      const decayHalfLife = Number(
        process.env.MEMORY_DECAY_HALF_LIFE_DAYS ?? 90,
      );

      if (!jsonOutput) {
        yield* Console.log(
          `Searching: "${query}"${ftsOnly ? " (FTS only)" : ""}\n`,
        );
      }

      let results: any;
      if (ftsOnly) {
        results = yield* db.ftsSearch(query, { limit, collection });
      } else {
        const embedding = yield* ollama.embed(query);
        results = yield* db.search(embedding, { limit, collection });
      }

      if (jsonOutput) {
        yield* Console.log(JSON.stringify(results, null, 2));
      } else if (results.length === 0) {
        yield* Console.log("No results found");
      } else {
        yield* Console.log(
          `Results (decay half-life: ${decayHalfLife} days):\n`,
        );
        let idx = 1;
        for (const r of results) {
          const ageDaysRounded = Math.round(r.ageDays);
          const decayPercent = Math.round(r.decayFactor * 100);
          const isStale = r.decayFactor < 0.5;

          const contentPreview = expand
            ? r.memory.content
            : `${r.memory.content.slice(0, 60).replace(/\n/g, " ")}${r.memory.content.length > 60 ? "..." : ""}`;
          yield* Console.log(
            `${idx}. [score: ${r.score.toFixed(2)}, age: ${ageDaysRounded}d, decay: ${decayPercent}%] ${contentPreview}`,
          );
          yield* Console.log(
            `   Collection: ${r.memory.collection} | ID: ${r.memory.id}`,
          );
          yield* Console.log(
            `   Raw: ${r.rawScore.toFixed(3)} → Final: ${r.score.toFixed(3)}`,
          );
          if (isStale) {
            yield* Console.log(
              `   ⚠️ Stale (${ageDaysRounded} days) - consider validating or removing`,
            );
          }
          yield* Console.log("");
          idx++;
        }
      }
      break;
    }

    case "list": {
      const collection = opts.collection as string | undefined;

      const memories = yield* db.list(collection);

      if (jsonOutput) {
        yield* Console.log(JSON.stringify(memories, null, 2));
      } else if (memories.length === 0) {
        yield* Console.log(
          collection
            ? `No memories in collection "${collection}"`
            : "No memories stored",
        );
      } else {
        yield* Console.log(`Memories: ${memories.length}\n`);
        for (const m of memories) {
          const tags = (m.metadata.tags as string[]) || [];
          const tagStr = tags.length ? ` [${tags.join(", ")}]` : "";
          yield* Console.log(`• ${m.id} (${m.collection})${tagStr}`);
          yield* Console.log(
            `  ${m.content.slice(0, 80).replace(/\n/g, " ")}${m.content.length > 80 ? "..." : ""}`,
          );
        }
      }
      break;
    }

    case "get": {
      const id = args[1];
      if (!id) {
        yield* Console.error("Error: ID required");
        process.exit(1);
      }

      const memory = yield* db.get(id);
      if (!memory) {
        yield* Console.error(`Not found: ${id}`);
        process.exit(1);
      }

      if (jsonOutput) {
        yield* Console.log(JSON.stringify(memory, null, 2));
      } else {
        yield* Console.log(`ID: ${memory.id}`);
        yield* Console.log(`Collection: ${memory.collection}`);
        yield* Console.log(`Created: ${memory.createdAt.toISOString()}`);
        yield* Console.log(`Metadata: ${JSON.stringify(memory.metadata)}`);
        yield* Console.log(`\nContent:\n${memory.content}`);
      }
      break;
    }

    case "delete": {
      const id = args[1];
      if (!id) {
        yield* Console.error("Error: ID required");
        process.exit(1);
      }

      const memory = yield* db.get(id);
      if (!memory) {
        yield* Console.error(`Not found: ${id}`);
        process.exit(1);
      }

      yield* db.delete(id);

      if (jsonOutput) {
        yield* Console.log(JSON.stringify({ deleted: id }));
      } else {
        yield* Console.log(`✓ Deleted: ${id}`);
      }
      break;
    }

    case "stats": {
      const stats = yield* db.getStats();
      const config = MemoryConfig.fromEnv();

      if (jsonOutput) {
        yield* Console.log(
          JSON.stringify({ ...stats, dataPath: config.dataPath }, null, 2),
        );
      } else {
        yield* Console.log(`Semantic Memory Stats`);
        yield* Console.log(`────────────────────`);
        yield* Console.log(`Memories:   ${stats.memories}`);
        yield* Console.log(`Embeddings: ${stats.embeddings}`);
        yield* Console.log(`Location:   ${config.dataPath}`);
      }
      break;
    }

    case "check": {
      yield* ollama.checkHealth();

      if (jsonOutput) {
        yield* Console.log(JSON.stringify({ status: "ok" }));
      } else {
        yield* Console.log("✓ Ollama is ready");
      }
      break;
    }

    case "validate": {
      const id = args[1];
      if (!id) {
        yield* Console.error("Error: ID required");
        process.exit(1);
      }

      const memory = yield* db.get(id);
      if (!memory) {
        yield* Console.error(`Not found: ${id}`);
        process.exit(1);
      }

      yield* db.validate(id);

      if (jsonOutput) {
        yield* Console.log(
          JSON.stringify({
            validated: id,
            timestamp: new Date().toISOString(),
          }),
        );
      } else {
        yield* Console.log(`✓ Validated: ${id}`);
        yield* Console.log(`  Decay timer reset to now`);
      }
      break;
    }

    default:
      yield* Console.error(`Unknown command: ${command}`);
      yield* Console.log(HELP);
      process.exit(1);
  }
});

// ============================================================================
// Run
// ============================================================================

const config = MemoryConfig.fromEnv();

// Check if this is a migrate command - handle before database initialization
const args = process.argv.slice(2);
const command = args[0];

if (command === "migrate") {
  // Run migration without database layer
  const migrateProgram = Effect.gen(function* () {
    const opts = parseArgs(args.slice(1));
    const jsonOutput = opts.json === true;
    const dataDir = `${config.dataPath}/memory`;

    if (opts.check) {
      const needs = yield* needsMigration(dataDir);
      if (jsonOutput) {
        yield* Console.log(JSON.stringify({ needsMigration: needs }));
      } else if (needs) {
        yield* Console.log(
          "⚠️  Migration needed: Database was created with PGlite 0.2.x (PostgreSQL 16)",
        );
        yield* Console.log("   Run: semantic-memory migrate");
      } else {
        yield* Console.log("✓ No migration needed");
      }
      return;
    }

    if (opts["generate-script"]) {
      const script = generateMigrationScript(dataDir);
      yield* Console.log(script);
      return;
    }

    if (opts.import) {
      const dumpPath = opts.import as string;
      if (!existsSync(dumpPath)) {
        yield* Console.error(`Error: Dump file not found: ${dumpPath}`);
        process.exit(1);
      }

      yield* Console.log(`Importing from ${dumpPath}...`);
      const result = yield* importMigrationDump(dataDir, dumpPath);

      if (jsonOutput) {
        yield* Console.log(JSON.stringify(result, null, 2));
      } else {
        yield* Console.log(`✓ Migration complete`);
        yield* Console.log(`  Memories: ${result.memoriesCount}`);
        yield* Console.log(`  Embeddings: ${result.embeddingsCount}`);
      }
      return;
    }

    // Default: attempt automatic migration
    const keepBackup = opts["no-backup"] !== true;
    yield* Console.log("Starting migration...\n");

    const result = yield* migrate(dataDir, { keepBackup });

    if (jsonOutput) {
      yield* Console.log(JSON.stringify(result, null, 2));
    } else {
      yield* Console.log(`\n✓ Migration complete`);
      yield* Console.log(`  Memories: ${result.memoriesCount}`);
      yield* Console.log(`  Embeddings: ${result.embeddingsCount}`);
      if (result.backupPath) {
        yield* Console.log(`  Backup: ${result.backupPath}`);
      }
    }
  });

  Effect.runPromise(
    migrateProgram.pipe(
      Effect.catchAll((error) =>
        Effect.gen(function* () {
          if ("reason" in error) {
            yield* Console.error(`Error: ${error.reason}`);
          } else {
            yield* Console.error(`Error: ${JSON.stringify(error)}`);
          }
          process.exit(1);
        }),
      ),
    ),
  );
} else {
  // Normal operation with database layer
  const ollamaLayer = makeOllamaLive(config);
  const databaseLayer = makeDatabaseLive({
    dbPath: `${config.dataPath}/memory.db`,
  });
  const serviceLayer = Layer.merge(ollamaLayer, databaseLayer);

  Effect.runPromise(
    program.pipe(
      Effect.provide(serviceLayer),
      Effect.catchAll((error) =>
        Effect.gen(function* () {
          // Check if this is a database initialization error that might need migration
          const errorStr = JSON.stringify(error);
          if (
            errorStr.includes("Unreachable code") ||
            errorStr.includes("pgl_backend")
          ) {
            yield* Console.error(
              "Error: Database initialization failed. This may be due to a PGlite version mismatch.",
            );
            yield* Console.error("Run: semantic-memory migrate --check");
          } else {
            yield* Console.error(
              `Error: ${error._tag}: ${JSON.stringify(error)}`,
            );
          }
          process.exit(1);
        }),
      ),
    ),
  );
}
