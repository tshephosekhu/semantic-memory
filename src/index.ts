/**
 * Semantic Memory - Public API
 *
 * Local semantic memory with PGlite + pgvector for AI agents.
 * Budget Qdrant that runs anywhere Bun runs.
 */

import { Effect, Layer } from "effect";
import { MemoryConfig } from "./types.js";
import {
  Database,
  makeDatabaseLive,
  DatabaseLive,
} from "./services/Database.js";
import { Ollama, makeOllamaLive } from "./services/Ollama.js";

// ============================================================================
// Re-exports - Types
// ============================================================================

export {
  Memory,
  SearchResult,
  MemoryConfig,
  SearchOptions,
  StoreOptions,
  OllamaError,
  DatabaseError,
  MemoryNotFoundError,
  ConfigError,
} from "./types.js";

// ============================================================================
// Re-exports - Services
// ============================================================================

export {
  Database,
  makeDatabaseLive,
  DatabaseLive,
} from "./services/Database.js";
export { Ollama, makeOllamaLive } from "./services/Ollama.js";

// ============================================================================
// Convenience Layer Composition
// ============================================================================

/**
 * Create a composed layer with all services configured.
 *
 * @example
 * ```ts
 * import { createMemoryService } from "semantic-memory";
 * import { Effect } from "effect";
 *
 * const program = Effect.gen(function* () {
 *   const db = yield* Database;
 *   const ollama = yield* Ollama;
 *   // ... use services
 * });
 *
 * Effect.runPromise(
 *   program.pipe(Effect.provide(createMemoryService()))
 * );
 * ```
 */
export function createMemoryService(config: Partial<MemoryConfig> = {}) {
  const fullConfig = new MemoryConfig({
    ...MemoryConfig.Default,
    ...config,
  });

  const ollamaLayer = makeOllamaLive(fullConfig);
  const databaseLayer = makeDatabaseLive({
    dbPath: `${fullConfig.dataPath}/memory.db`,
  });

  return Layer.merge(ollamaLayer, databaseLayer);
}

/**
 * Default layer using environment configuration.
 * Reads from:
 * - SEMANTIC_MEMORY_PATH (default: ~/.semantic-memory)
 * - OLLAMA_MODEL (default: mxbai-embed-large)
 * - OLLAMA_HOST (default: http://localhost:11434)
 */
export const MemoryServiceLive = createMemoryService(MemoryConfig.fromEnv());
