#!/usr/bin/env bun
/**
 * Semantic Memory CLI
 *
 * Store and retrieve memories with semantic search.
 */

import { Effect, Console, Layer } from "effect";
import { randomUUID } from "crypto";
import { Database, makeDatabaseLive } from "./services/Database.js";
import { Ollama, makeOllamaLive } from "./services/Ollama.js";
import { MemoryConfig } from "./types.js";

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

  list                    List all memories
    --collection <name>   Filter by collection

  get <id>                Get a memory by ID

  delete <id>             Delete a memory by ID

  stats                   Show memory statistics

  check                   Verify Ollama is running

Options:
  --help, -h              Show this help
  --json                  Output as JSON

Examples:
  semantic-memory store "Meeting notes from standup" --tags "meetings,work"
  semantic-memory find "what did we discuss in standup" --limit 5
  semantic-memory list --collection work
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

      if (!jsonOutput) {
        yield* Console.log(
          `Searching: "${query}"${ftsOnly ? " (FTS only)" : ""}\n`,
        );
      }

      let results;
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
        for (const r of results) {
          yield* Console.log(
            `[${r.score.toFixed(3)}] ${r.memory.collection} - ${r.memory.id.slice(0, 8)}...`,
          );
          yield* Console.log(
            `  ${r.memory.content.slice(0, 200).replace(/\n/g, " ")}${r.memory.content.length > 200 ? "..." : ""}`,
          );
          yield* Console.log("");
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
          yield* Console.log(
            `• ${m.id.slice(0, 8)}... (${m.collection})${tagStr}`,
          );
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
        yield* Console.error(`Error: ${error._tag}: ${JSON.stringify(error)}`);
        process.exit(1);
      }),
    ),
  ),
);
