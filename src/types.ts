/**
 * Semantic Memory Types
 *
 * Domain models, configuration, and errors for the semantic memory system.
 * Uses Effect Schema for runtime validation and type safety.
 */

import { Schema } from "effect";

// ============================================================================
// Domain Models
// ============================================================================

/**
 * A memory unit stored in the vector database.
 * Contains content, embeddings (stored separately), and flexible metadata.
 */
export class Memory extends Schema.Class<Memory>("Memory")({
  id: Schema.String,
  content: Schema.String,
  metadata: Schema.Record({ key: Schema.String, value: Schema.Unknown }),
  collection: Schema.String,
  createdAt: Schema.Date,
}) {}

/**
 * Result from a semantic or hybrid search operation.
 * Includes similarity score and match type for ranking/filtering.
 */
export class SearchResult extends Schema.Class<SearchResult>("SearchResult")({
  memoryId: Schema.String,
  content: Schema.String,
  metadata: Schema.Record({ key: Schema.String, value: Schema.Unknown }),
  score: Schema.Number,
  matchType: Schema.Literal("vector", "fts", "hybrid"),
}) {}

// ============================================================================
// Configuration
// ============================================================================

/**
 * Configuration for the semantic memory system.
 * All values can be overridden via environment variables.
 */
export class MemoryConfig extends Schema.Class<MemoryConfig>("MemoryConfig")({
  /** Path to the data directory (default: ~/.semantic-memory) */
  dataPath: Schema.String,
  /** Ollama embedding model to use */
  ollamaModel: Schema.String,
  /** Ollama API host */
  ollamaHost: Schema.String,
  /** MCP tool description for store operation */
  toolStoreDescription: Schema.String,
  /** MCP tool description for find operation */
  toolFindDescription: Schema.String,
  /** Default collection name for memories */
  defaultCollection: Schema.String,
}) {
  static readonly Default = new MemoryConfig({
    dataPath: `${process.env.HOME}/.semantic-memory`,
    ollamaModel: "mxbai-embed-large",
    ollamaHost: "http://localhost:11434",
    toolStoreDescription:
      "Store a memory with semantic embeddings for later retrieval",
    toolFindDescription: "Find memories using semantic similarity search",
    defaultCollection: "default",
  });

  /**
   * Create configuration from environment variables.
   * Falls back to defaults for any unset variables.
   */
  static fromEnv(): MemoryConfig {
    return new MemoryConfig({
      dataPath:
        process.env.SEMANTIC_MEMORY_PATH ||
        `${process.env.HOME}/.semantic-memory`,
      ollamaModel: process.env.OLLAMA_MODEL || "mxbai-embed-large",
      ollamaHost: process.env.OLLAMA_HOST || "http://localhost:11434",
      toolStoreDescription:
        process.env.TOOL_STORE_DESCRIPTION ||
        "Store a memory with semantic embeddings for later retrieval",
      toolFindDescription:
        process.env.TOOL_FIND_DESCRIPTION ||
        "Find memories using semantic similarity search",
      defaultCollection: process.env.COLLECTION_NAME || "default",
    });
  }
}

// ============================================================================
// Search Options
// ============================================================================

/**
 * Options for semantic search queries.
 * All fields are optional with sensible defaults.
 */
export class SearchOptions extends Schema.Class<SearchOptions>("SearchOptions")(
  {
    /** Maximum number of results to return */
    limit: Schema.optionalWith(Schema.Number, { default: () => 10 }),
    /** Minimum similarity threshold (0-1) */
    threshold: Schema.optionalWith(Schema.Number, { default: () => 0.3 }),
    /** Filter by collection name */
    collection: Schema.optional(Schema.String),
    /** Filter by metadata tags */
    tags: Schema.optional(Schema.Array(Schema.String)),
  },
) {}

/**
 * Options for storing a new memory.
 */
export class StoreOptions extends Schema.Class<StoreOptions>("StoreOptions")({
  /** Collection to store the memory in */
  collection: Schema.optional(Schema.String),
  /** Metadata tags for filtering */
  tags: Schema.optional(Schema.Array(Schema.String)),
  /** Additional metadata key-value pairs */
  metadata: Schema.optional(
    Schema.Record({ key: Schema.String, value: Schema.Unknown }),
  ),
}) {}

// ============================================================================
// Errors
// ============================================================================

/**
 * Error communicating with or processing via Ollama.
 */
export class OllamaError extends Schema.TaggedError<OllamaError>()(
  "OllamaError",
  { reason: Schema.String },
) {}

/**
 * Database operation failure (PGlite/pgvector).
 */
export class DatabaseError extends Schema.TaggedError<DatabaseError>()(
  "DatabaseError",
  { reason: Schema.String },
) {}

/**
 * Requested memory was not found.
 */
export class MemoryNotFoundError extends Schema.TaggedError<MemoryNotFoundError>()(
  "MemoryNotFoundError",
  { id: Schema.String },
) {}

/**
 * Configuration validation error.
 */
export class ConfigError extends Schema.TaggedError<ConfigError>()(
  "ConfigError",
  { reason: Schema.String },
) {}
