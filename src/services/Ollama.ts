/**
 * Ollama Embedding Service
 *
 * Provides embedding generation via local Ollama server.
 * Uses Effect-TS patterns: Context.Tag for DI, Layer for instantiation.
 */

import {
  Effect,
  Context,
  Layer,
  Schedule,
  Duration,
  Chunk,
  Stream,
} from "effect";
import type { MemoryConfig } from "../types.js";
import { OllamaError } from "../types.js";

// ============================================================================
// Service Definition
// ============================================================================

/**
 * Ollama service for generating embeddings from text.
 *
 * @example
 * ```ts
 * const embedding = yield* Ollama.embed("hello world");
 * // => number[] (768 dims for nomic-embed-text)
 * ```
 */
export class Ollama extends Context.Tag("semantic-memory/Ollama")<
  Ollama,
  {
    /** Generate embedding for a single text */
    readonly embed: (text: string) => Effect.Effect<number[], OllamaError>;
    /** Generate embeddings for multiple texts with controlled concurrency */
    readonly embedBatch: (
      texts: string[],
      concurrency?: number,
    ) => Effect.Effect<number[][], OllamaError>;
    /** Verify Ollama is running and model is available */
    readonly checkHealth: () => Effect.Effect<void, OllamaError>;
  }
>() {}

// ============================================================================
// Implementation
// ============================================================================

interface OllamaEmbeddingResponse {
  embedding: number[];
}

interface OllamaTagsResponse {
  models: Array<{ name: string }>;
}

/**
 * Create Ollama service layer from config.
 *
 * @param config - Memory configuration with ollamaHost and ollamaModel
 * @returns Layer providing the Ollama service
 */
export const makeOllamaLive = (config: MemoryConfig) =>
  Layer.succeed(
    Ollama,
    (() => {
      const embedSingle = (
        text: string,
      ): Effect.Effect<number[], OllamaError> =>
        Effect.gen(function* () {
          const response = yield* Effect.tryPromise({
            try: () =>
              fetch(`${config.ollamaHost}/api/embeddings`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  model: config.ollamaModel,
                  prompt: text,
                }),
              }),
            catch: (e) =>
              new OllamaError({ reason: `Connection failed: ${e}` }),
          });

          if (!response.ok) {
            const error = yield* Effect.tryPromise({
              try: () => response.text(),
              catch: () =>
                new OllamaError({ reason: "Failed to read error response" }),
            });
            return yield* Effect.fail(new OllamaError({ reason: error }));
          }

          const data = yield* Effect.tryPromise({
            try: () => response.json() as Promise<OllamaEmbeddingResponse>,
            catch: () => new OllamaError({ reason: "Invalid JSON response" }),
          });

          return data.embedding;
        }).pipe(
          // Retry with exponential backoff on transient failures
          // 100ms -> 200ms -> 400ms (3 attempts total)
          Effect.retry(
            Schedule.exponential(Duration.millis(100)).pipe(
              Schedule.compose(Schedule.recurs(3)),
            ),
          ),
        );

      return {
        embed: embedSingle,

        embedBatch: (texts: string[], concurrency = 5) =>
          Stream.fromIterable(texts).pipe(
            Stream.mapEffect(embedSingle, { concurrency }),
            Stream.runCollect,
            Effect.map(Chunk.toArray),
          ),

        checkHealth: () =>
          Effect.gen(function* () {
            const response = yield* Effect.tryPromise({
              try: () => fetch(`${config.ollamaHost}/api/tags`),
              catch: () =>
                new OllamaError({
                  reason: `Cannot connect to Ollama at ${config.ollamaHost}`,
                }),
            });

            if (!response.ok) {
              return yield* Effect.fail(
                new OllamaError({ reason: "Ollama not responding" }),
              );
            }

            const data = yield* Effect.tryPromise({
              try: () => response.json() as Promise<OllamaTagsResponse>,
              catch: () =>
                new OllamaError({ reason: "Invalid response from Ollama" }),
            });

            const hasModel = data.models.some(
              (m) =>
                m.name === config.ollamaModel ||
                m.name.startsWith(`${config.ollamaModel}:`),
            );

            if (!hasModel) {
              return yield* Effect.fail(
                new OllamaError({
                  reason: `Model ${config.ollamaModel} not found. Run: ollama pull ${config.ollamaModel}`,
                }),
              );
            }
          }),
      };
    })(),
  );
