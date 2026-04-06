/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.codecs.lsmvec;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.SuppressForbidden;

@SuppressForbidden(reason = "Benchmark uses system time and printf")
public class TestLsmVecBenchmark extends LuceneTestCase {

  static final int NUM_DOCS = 10000;
  static final int NUM_QUERIES = 1000;
  static final int DIMENSIONS = 128;
  static final int TOP_K = 10;

  // NOTE: Remove @Ignore when running manually if needed, or leave it and run via
  // specific test
  // command.
  // Actually, we WANT this to run right now via gradle, so no @Ignore!

  public void testBenchmark() throws Exception {
    System.out.println("Generating clustered normalized vectors...");
    Random rand = new Random(42);

    int numCentroids = 100;
    float[][] centroids = new float[numCentroids][DIMENSIONS];
    for (int c = 0; c < numCentroids; c++) {
      for (int d = 0; d < DIMENSIONS; d++) {
        centroids[c][d] = rand.nextFloat() * 2 - 1f;
      }
    }

    float[][] indexVectors = new float[NUM_DOCS][DIMENSIONS];
    for (int i = 0; i < NUM_DOCS; i++) {
      int centroidIdx = rand.nextInt(numCentroids);
      float norm = 0;
      for (int d = 0; d < DIMENSIONS; d++) {
        float val = centroids[centroidIdx][d] + (float) (rand.nextGaussian() * 0.1);
        indexVectors[i][d] = val;
        norm += val * val;
      }
      norm = (float) Math.sqrt(norm);
      for (int d = 0; d < DIMENSIONS; d++) {
        indexVectors[i][d] /= norm;
      }
    }

    float[][] queryVectors = new float[NUM_QUERIES][DIMENSIONS];
    for (int i = 0; i < NUM_QUERIES; i++) {
      int centroidIdx = rand.nextInt(numCentroids);
      float norm = 0;
      for (int d = 0; d < DIMENSIONS; d++) {
        float val = centroids[centroidIdx][d] + (float) (rand.nextGaussian() * 0.1);
        queryVectors[i][d] = val;
        norm += val * val;
      }
      norm = (float) Math.sqrt(norm);
      for (int d = 0; d < DIMENSIONS; d++) {
        queryVectors[i][d] /= norm;
      }
    }

    System.out.println("Calculating exact ground truth...");
    int[][] groundTruth = new int[NUM_QUERIES][TOP_K];
    for (int q = 0; q < NUM_QUERIES; q++) {
      float[] query = queryVectors[q];

      class DocScore {
        final int docId;
        final float score;

        DocScore(int id, float s) {
          this.docId = id;
          this.score = s;
        }
      }

      DocScore[] scores = new DocScore[NUM_DOCS];
      for (int d = 0; d < NUM_DOCS; d++) {
        float distance = 0;
        for (int i = 0; i < DIMENSIONS; i++) {
          float diff = query[i] - indexVectors[d][i];
          distance += diff * diff;
        }
        // Lucene uses 1 / (1 + distance) for Euclidean score
        float score = 1f / (1f + distance);
        scores[d] = new DocScore(d, score);
      }

      Arrays.sort(scores, Comparator.comparingDouble((DocScore ds) -> ds.score).reversed());
      for (int k = 0; k < TOP_K; k++) {
        groundTruth[q][k] = scores[k].docId;
      }
    }

    record TestCombo(String name, KnnVectorsFormat format) {
    }

    TestCombo[] combos =
        new TestCombo[] {
          new TestCombo("LsmVec", new LsmVecVectorsFormat()),
          new TestCombo(
              "Lucene104", new org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat())
        };

    System.out.println("Warming up JIT & merging code paths... (JMH API equivalents)");
    int warmups = 6;
    for (int warmupId = 0; warmupId < warmups + 1; warmupId++) {
      boolean isWarmup = (warmupId != warmups);
      if (!isWarmup) {
        System.out.println("\n--- Benchmark Results ---");
        System.out.printf(
            "%-15s | %-12s | %-12s | %-10s | %-12s | %-10s\n",
            "Codec", "Index (ms)", "Merge (ms)", "Size (KB)", "Query (ms)", "Recall@10");
        System.out.println(
            "-----------------------------------------------------------------------------------");
      }

      for (TestCombo combo : combos) {
        KnnVectorsFormat format = combo.format;
        Path tempDir = createTempDir(combo.name);
        long indexTimeMs;
        long mergeTimeMs;
        long sizeKB;
        long queryTimeMs;
        double recall;

        try (Directory dir = FSDirectory.open(tempDir)) {
          IndexWriterConfig iwc = newIndexWriterConfig();
          iwc.setUseCompoundFile(false);
          iwc.setCodec(org.apache.lucene.tests.util.TestUtil.alwaysKnnVectorsFormat(format));
          org.apache.lucene.index.LogDocMergePolicy mergePolicy =
              new org.apache.lucene.index.LogDocMergePolicy();
          mergePolicy.setMergeFactor(100000); // maintain rigid sequence
          iwc.setMergePolicy(mergePolicy);
          iwc.setMergeScheduler(new org.apache.lucene.index.SerialMergeScheduler());
          // Ensure some merging happens naturally, but we also force merge

          long startIdx = System.currentTimeMillis();
          try (IndexWriter writer = new IndexWriter(dir, iwc)) {
            int halfDocs = NUM_DOCS / 2;
            for (int i = 0; i < halfDocs; i++) {
              Document doc = new Document();
              doc.add(
                  new KnnFloatVectorField(
                      "vector", indexVectors[i], VectorSimilarityFunction.EUCLIDEAN));
              writer.addDocument(doc);
            }
            writer.commit();

            for (int i = halfDocs; i < NUM_DOCS; i++) {
              Document doc = new Document();
              doc.add(
                  new KnnFloatVectorField(
                      "vector", indexVectors[i], VectorSimilarityFunction.EUCLIDEAN));
              writer.addDocument(doc);
            }
            writer.commit();
            indexTimeMs = System.currentTimeMillis() - startIdx;

            long startMerge = System.currentTimeMillis();
            writer.forceMerge(1);
            writer.commit();
            mergeTimeMs = System.currentTimeMillis() - startMerge;
          }

          // Calculate size
          long totalBytes = 0;
          for (String file : dir.listAll()) {
            totalBytes += dir.fileLength(file);
          }
          sizeKB = totalBytes / 1024;

          try (IndexReader reader = DirectoryReader.open(dir)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            long startQuery = System.currentTimeMillis();
            int totalRecalled = 0;

            for (int q = 0; q < NUM_QUERIES; q++) {
              KnnFloatVectorQuery kvq = new KnnFloatVectorQuery("vector", queryVectors[q], TOP_K);
              TopDocs docs = searcher.search(kvq, TOP_K);

              Set<Integer> truth = new HashSet<>();
              for (int k = 0; k < TOP_K; k++) {
                truth.add(groundTruth[q][k]);
              }

              int matchCount = 0;
              for (int i = 0; i < docs.scoreDocs.length; i++) {
                if (truth.contains(docs.scoreDocs[i].doc)) {
                  matchCount++;
                }
              }
              totalRecalled += matchCount;
            }

            queryTimeMs = System.currentTimeMillis() - startQuery;
            recall = totalRecalled / (double) (NUM_QUERIES * TOP_K) * 100.0;
          }
        }

        if (!isWarmup) {
          System.out.printf(
              "%-15s | %-12d | %-12d | %-10d | %-12d | %-5.2f%%\n",
              combo.name, indexTimeMs, mergeTimeMs, sizeKB, queryTimeMs, recall);
        }
      }
    }
  }
}
