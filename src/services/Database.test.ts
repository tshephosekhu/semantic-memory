/**
 * Database.ts Tests
 *
 * Testing strategy:
 * - Single shared database instance (fast init)
 * - Unique collection per test (isolation without DB overhead)
 * - Effect.runPromise for async Effect execution
 * - Cover CRUD, vector search, FTS, collection filtering, decay logic
 */

import { beforeAll, afterAll, describe, expect, test } from "bun:test";
import { Effect } from "effect";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Database, makeDatabaseLive, type Memory } from "./Database";

// ============================================================================
// Test Fixtures
// ============================================================================

/** Shared database path - created once for all tests */
let sharedDbPath: string;
let sharedDbLayer: ReturnType<typeof makeDatabaseLive>;

/** Generate unique collection name per test */
let testCounter = 0;
function uniqueCollection(): string {
  return `test-${Date.now()}-${++testCounter}`;
}

/** Create test memory with minimal required fields */
function makeMemory(
  overrides: Partial<Memory> = {},
  collection?: string
): Memory {
  return {
    id: `mem-${Date.now()}-${Math.random()}`,
    content: "Test memory content",
    metadata: {},
    collection: collection ?? "default",
    createdAt: new Date(),
    ...overrides,
  };
}

/** Create dummy embedding vector (1024 dimensions for mxbai-embed-large) */
function makeEmbedding(seed = 1.0): number[] {
  return Array.from({ length: 1024 }, (_, i) => Math.sin(seed + i * 0.1));
}

// ============================================================================
// Test Harness - Single DB for all tests
// ============================================================================

beforeAll(() => {
  const tempDir = mkdtempSync(join(tmpdir(), "semantic-memory-test-"));
  sharedDbPath = join(tempDir, "test.db");
  sharedDbLayer = makeDatabaseLive({ dbPath: sharedDbPath });
});

afterAll(() => {
  // Cleanup temp database
  const dbDir = sharedDbPath.replace(".db", "");
  try {
    rmSync(dbDir, { recursive: true, force: true });
  } catch {
    // ignore cleanup errors
  }
});

describe("Database", () => {
  // ==========================================================================
  // CRUD Operations
  // ==========================================================================

  describe("store", () => {
    test("stores memory with embedding", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({ collection: coll });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);
        const retrieved = yield* db.get(memory.id);
        return retrieved;
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result).not.toBeNull();
      expect(result?.id).toBe(memory.id);
      expect(result?.content).toBe(memory.content);
      expect(result?.collection).toBe(coll);
    });

    test("updates existing memory on conflict", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({
        id: `mem-update-${coll}`,
        content: "Original",
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Store original
        yield* db.store(memory, embedding);

        // Update with new content
        const updated = { ...memory, content: "Updated" };
        yield* db.store(updated, embedding);

        return yield* db.get(memory.id);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result?.content).toBe("Updated");
    });

    test("stores metadata as JSON", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({
        metadata: { tags: ["test", "important"], priority: 1 },
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result?.metadata).toEqual({
        tags: ["test", "important"],
        priority: 1,
      });
    });

    test("handles custom collection", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({ collection: coll });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result?.collection).toBe(coll);
    });
  });

  describe("get", () => {
    test("returns null for non-existent memory", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;
        return yield* db.get("non-existent-id-xyz");
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toBeNull();
    });

    test("retrieves memory with all fields", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({
        content: "Full memory",
        metadata: { key: "value" },
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result).not.toBeNull();
      expect(result?.id).toBe(memory.id);
      expect(result?.content).toBe("Full memory");
      expect(result?.metadata).toEqual({ key: "value" });
      expect(result?.collection).toBe(coll);
      expect(result?.createdAt).toBeInstanceOf(Date);
    });
  });

  describe("list", () => {
    test("returns empty array for empty collection", async () => {
      const coll = uniqueCollection();
      const program = Effect.gen(function* () {
        const db = yield* Database;
        return yield* db.list(coll);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toEqual([]);
    });

    test("lists all memories in collection ordered by creation date", async () => {
      const coll = uniqueCollection();
      const now = new Date();
      const earlier = new Date(now.getTime() - 1000);

      const mem1 = makeMemory({
        id: `mem-1-${coll}`,
        createdAt: earlier,
        collection: coll,
      });
      const mem2 = makeMemory({
        id: `mem-2-${coll}`,
        createdAt: now,
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(mem1, embedding);
        yield* db.store(mem2, embedding);
        return yield* db.list(coll);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result.length).toBe(2);
      // Most recent first
      expect(result[0].id).toBe(`mem-2-${coll}`);
      expect(result[1].id).toBe(`mem-1-${coll}`);
    });

    test("filters by collection", async () => {
      const coll1 = uniqueCollection();
      const coll2 = uniqueCollection();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(makeMemory({ collection: coll1 }), embedding);
        yield* db.store(makeMemory({ collection: coll2 }), embedding);
        yield* db.store(makeMemory({ collection: coll1 }), embedding);

        return yield* db.list(coll1);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result.length).toBe(2);
      expect(result.every((m) => m.collection === coll1)).toBe(true);
    });
  });

  describe("delete", () => {
    test("deletes memory by id", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({ collection: coll });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        // Verify exists
        const before = yield* db.get(memory.id);
        expect(before).not.toBeNull();

        // Delete
        yield* db.delete(memory.id);

        // Verify gone
        return yield* db.get(memory.id);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toBeNull();
    });

    test("deleting non-existent memory succeeds silently", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.delete("non-existent-delete-test");
        return "success";
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toBe("success");
    });
  });

  // ==========================================================================
  // Vector Search
  // ==========================================================================

  describe("search (vector)", () => {
    test("finds similar memories by embedding", async () => {
      const coll = uniqueCollection();
      const mem1 = makeMemory({
        content: "Machine learning basics",
        collection: coll,
      });
      const mem2 = makeMemory({ content: "Cooking recipes", collection: coll });
      const embedding1 = makeEmbedding(1.0);
      const embedding2 = makeEmbedding(100.0);

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(mem1, embedding1);
        yield* db.store(mem2, embedding2);

        return yield* db.search(embedding1, {
          limit: 10,
          threshold: 0.3,
          collection: coll,
        });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].memory.id).toBe(mem1.id);
      expect(results[0].score).toBeGreaterThan(0.9);
      expect(results[0].matchType).toBe("vector");
    });

    test("respects similarity threshold", async () => {
      const coll = uniqueCollection();
      const mem1 = makeMemory({ content: "Test", collection: coll });
      const embedding1 = makeEmbedding(1.0);
      const queryEmbedding = makeEmbedding(500.0);

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(mem1, embedding1);

        return yield* db.search(queryEmbedding, {
          threshold: 0.9,
          collection: coll,
        });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(0);
    });

    test("respects limit", async () => {
      const coll = uniqueCollection();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        for (let i = 0; i < 5; i++) {
          yield* db.store(
            makeMemory({ id: `mem-limit-${coll}-${i}`, collection: coll }),
            embedding
          );
        }

        return yield* db.search(embedding, { limit: 2, collection: coll });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(2);
    });

    test("filters by collection", async () => {
      const coll1 = uniqueCollection();
      const coll2 = uniqueCollection();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(makeMemory({ collection: coll1 }), embedding);
        yield* db.store(makeMemory({ collection: coll2 }), embedding);

        return yield* db.search(embedding, { collection: coll1 });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      expect(results[0].memory.collection).toBe(coll1);
    });

    test("includes decay information", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({ collection: coll });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.search(embedding, { limit: 1, collection: coll });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      const result = results[0];

      expect(result.ageDays).toBeGreaterThanOrEqual(0);
      expect(result.ageDays).toBeLessThan(1);
      expect(result.decayFactor).toBeGreaterThan(0.99);
      expect(result.rawScore).toBeGreaterThan(0);
      expect(result.score).toBeGreaterThan(0);
    });

    test("applies decay to score over time", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({
        createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.search(embedding, {
          limit: 1,
          threshold: 0.0,
          collection: coll,
        });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      const result = results[0];

      expect(result.ageDays).toBeGreaterThan(89);
      expect(result.decayFactor).toBeLessThan(0.6);
      expect(result.decayFactor).toBeGreaterThan(0.4);
      expect(result.score).toBeLessThan(result.rawScore);
    });
  });

  // ==========================================================================
  // Full-Text Search
  // ==========================================================================

  describe("ftsSearch", () => {
    test("finds memories by text content", async () => {
      const coll = uniqueCollection();
      const mem1 = makeMemory({
        content: "PostgreSQL database optimization",
        collection: coll,
      });
      const mem2 = makeMemory({
        content: "React component patterns",
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(mem1, embedding);
        yield* db.store(mem2, embedding);

        return yield* db.ftsSearch("PostgreSQL", {
          limit: 10,
          collection: coll,
        });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].memory.id).toBe(mem1.id);
      expect(results[0].matchType).toBe("fts");
    });

    test("handles multi-word queries", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({
        content: "Database optimization techniques",
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.ftsSearch("database optimization", {
          limit: 10,
          collection: coll,
        });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      expect(results[0].memory.id).toBe(memory.id);
    });

    test("respects limit", async () => {
      const coll = uniqueCollection();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        for (let i = 0; i < 5; i++) {
          yield* db.store(
            makeMemory({
              id: `mem-fts-${coll}-${i}`,
              content: "test content fts",
              collection: coll,
            }),
            embedding
          );
        }

        return yield* db.ftsSearch("test", { limit: 2, collection: coll });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(2);
    });

    test("filters by collection", async () => {
      const coll1 = uniqueCollection();
      const coll2 = uniqueCollection();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(
          makeMemory({ collection: coll1, content: "searchable item" }),
          embedding
        );
        yield* db.store(
          makeMemory({ collection: coll2, content: "searchable item" }),
          embedding
        );

        return yield* db.ftsSearch("searchable", { collection: coll1 });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      expect(results[0].memory.collection).toBe(coll1);
    });

    test("returns empty array for no matches", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({
        content: "PostgreSQL database",
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.ftsSearch("nonexistent search term xyz", {
          limit: 10,
          collection: coll,
        });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(0);
    });

    test("includes decay information", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({
        content: "test content decay",
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        return yield* db.ftsSearch("decay", { limit: 1, collection: coll });
      }).pipe(Effect.provide(sharedDbLayer));

      const results = await Effect.runPromise(program);

      expect(results.length).toBe(1);
      const result = results[0];

      expect(result.ageDays).toBeGreaterThanOrEqual(0);
      expect(result.decayFactor).toBeGreaterThan(0.99);
      expect(result.rawScore).toBeGreaterThan(0);
      expect(result.score).toBeGreaterThan(0);
    });
  });

  // ==========================================================================
  // Validation (Decay Reset)
  // ==========================================================================

  describe("validate", () => {
    test("sets lastValidatedAt timestamp", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({ collection: coll });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, embedding);

        const before = yield* db.get(memory.id);
        expect(before?.lastValidatedAt).toBeUndefined();

        yield* db.validate(memory.id);

        return yield* db.get(memory.id);
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result?.lastValidatedAt).toBeInstanceOf(Date);
      expect(result?.lastValidatedAt).toBeDefined();
    });

    test("validation resets decay for search", async () => {
      const coll = uniqueCollection();
      const oldMemory = makeMemory({
        createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(oldMemory, embedding);

        const beforeValidation = yield* db.search(embedding, {
          limit: 1,
          threshold: 0.0,
          collection: coll,
        });
        const decayBefore = beforeValidation[0].decayFactor;

        yield* db.validate(oldMemory.id);

        const afterValidation = yield* db.search(embedding, {
          limit: 1,
          threshold: 0.0,
          collection: coll,
        });

        return {
          decayBefore,
          decayAfter: afterValidation[0].decayFactor,
        };
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result.decayAfter).toBeGreaterThan(result.decayBefore);
      expect(result.decayAfter).toBeGreaterThan(0.99);
    });

    test("validation affects FTS search decay", async () => {
      const coll = uniqueCollection();
      const oldMemory = makeMemory({
        content: "validate fts test",
        createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000),
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(oldMemory, embedding);

        const beforeValidation = yield* db.ftsSearch("validate", {
          limit: 1,
          collection: coll,
        });
        const decayBefore = beforeValidation[0].decayFactor;

        yield* db.validate(oldMemory.id);

        const afterValidation = yield* db.ftsSearch("validate", {
          limit: 1,
          collection: coll,
        });

        return {
          decayBefore,
          decayAfter: afterValidation[0].decayFactor,
        };
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result.decayAfter).toBeGreaterThan(result.decayBefore);
      expect(result.decayAfter).toBeGreaterThan(0.99);
    });
  });

  // ==========================================================================
  // Statistics
  // ==========================================================================

  describe("getStats", () => {
    test("counts memories and embeddings", async () => {
      const coll = uniqueCollection();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        const before = yield* db.getStats();

        yield* db.store(
          makeMemory({ id: `mem-stats-1-${coll}`, collection: coll }),
          embedding
        );
        yield* db.store(
          makeMemory({ id: `mem-stats-2-${coll}`, collection: coll }),
          embedding
        );

        const after = yield* db.getStats();

        return { before, after };
      }).pipe(Effect.provide(sharedDbLayer));

      const { before, after } = await Effect.runPromise(program);

      // Should have 2 more than before
      expect(after.memories).toBe(before.memories + 2);
      expect(after.embeddings).toBe(before.embeddings + 2);
    });
  });

  // ==========================================================================
  // Error Cases
  // ==========================================================================

  describe("error handling", () => {
    test("store fails with invalid embedding dimension", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({ collection: coll });
      const invalidEmbedding = [1, 2, 3];

      const program = Effect.gen(function* () {
        const db = yield* Database;
        yield* db.store(memory, invalidEmbedding);
        return "should not reach here";
      }).pipe(Effect.provide(sharedDbLayer));

      try {
        await Effect.runPromise(program);
        expect.unreachable("Expected store to fail with invalid embedding");
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    test("search fails with invalid embedding dimension", async () => {
      const coll = uniqueCollection();
      const invalidEmbedding = [1, 2, 3];

      const program = Effect.gen(function* () {
        const db = yield* Database;
        return yield* db.search(invalidEmbedding, { collection: coll });
      }).pipe(Effect.provide(sharedDbLayer));

      try {
        await Effect.runPromise(program);
        expect.unreachable("Expected search to fail with invalid embedding");
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  // ==========================================================================
  // Integration Scenarios
  // ==========================================================================

  describe("integration scenarios", () => {
    test("full workflow: store, search, validate, delete", async () => {
      const coll = uniqueCollection();
      const memory = makeMemory({
        content: "Integration test memory",
        collection: coll,
      });
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        yield* db.store(memory, embedding);

        const searchResults = yield* db.search(embedding, {
          limit: 1,
          collection: coll,
        });
        expect(searchResults.length).toBe(1);
        expect(searchResults[0].memory.id).toBe(memory.id);

        yield* db.validate(memory.id);
        const validated = yield* db.get(memory.id);
        expect(validated?.lastValidatedAt).toBeDefined();

        yield* db.delete(memory.id);
        const deleted = yield* db.get(memory.id);
        expect(deleted).toBeNull();

        return "success";
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result).toBe("success");
    });

    test("multiple collections work independently", async () => {
      const coll1 = uniqueCollection();
      const coll2 = uniqueCollection();
      const embedding = makeEmbedding();

      const program = Effect.gen(function* () {
        const db = yield* Database;

        yield* db.store(makeMemory({ collection: coll1 }), embedding);
        yield* db.store(makeMemory({ collection: coll1 }), embedding);
        yield* db.store(makeMemory({ collection: coll2 }), embedding);

        const list1 = yield* db.list(coll1);
        const list2 = yield* db.list(coll2);

        return { list1, list2 };
      }).pipe(Effect.provide(sharedDbLayer));

      const result = await Effect.runPromise(program);

      expect(result.list1.length).toBe(2);
      expect(result.list2.length).toBe(1);
    });
  });
});
