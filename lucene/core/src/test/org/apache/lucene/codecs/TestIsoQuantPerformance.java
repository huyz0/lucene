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
package org.apache.lucene.codecs;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import org.apache.lucene.codecs.isoquant.IsoEncoding;
import org.apache.lucene.codecs.isoquant.IsoQuantHnswVectorsFormat;
import org.apache.lucene.codecs.isoquant.IsoVariant;
import org.apache.lucene.codecs.isoquant.IsoCompress;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LogDocMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;

public class TestIsoQuantPerformance extends LuceneTestCase {

  private int numDocs;
  private int dimension;
  private int numQueries;
  private static final int TOP_K = 10;

  enum Config {
    BASELINE_FP32,
    BASELINE_INT4,
    BASELINE_INT8,
    ISOQUANT_4BIT,
    ISOQUANT_8BIT,
    ISOQUANT_4BIT_LZ4,
    ISOQUANT_8BIT_LZ4
  }

  static class Result {
    double recall;
    long indexMs;
    long mergeMs;
    long searchMs;
    long diskSizeBytes;

    Result(double r, long i, long m, long s, long b) {
      recall = r;
      indexMs = i;
      mergeMs = m;
      searchMs = s;
      diskSizeBytes = b;
    }
  }

  private float[][] dataset;
  private float[][] queries;
  private int[][] groundtruth;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    try (InputStream baseIs =
            TestIsoQuantPerformance.class.getResourceAsStream("siftsmall/siftsmall_base.fvecs");
        InputStream queryIs =
            TestIsoQuantPerformance.class.getResourceAsStream("siftsmall/siftsmall_query.fvecs");
        InputStream gtIs =
            TestIsoQuantPerformance.class.getResourceAsStream("siftsmall/siftsmall_groundtruth.ivecs")) {
      if (baseIs == null || queryIs == null || gtIs == null) {
        throw new RuntimeException("Could not load true SIFT10k dataset from test resources: missing files.");
      }
      System.out.println("Loading true SIFT10k dataset from test resources...");
      dataset = readFvecs(baseIs);
      queries = readFvecs(queryIs);
      groundtruth = readIvecs(gtIs);
      numDocs = dataset.length;
      dimension = dataset[0].length;
      numQueries = queries.length;
    }
  }

  private float[][] readFvecs(InputStream is) throws Exception {
    byte[] bytes = is.readAllBytes();
    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    int numVecs = bytes.length / (4 + 128 * 4);
    float[][] ret = new float[numVecs][128];
    for (int i = 0; i < numVecs; i++) {
      int dim = buf.getInt();
      if (dim != 128) throw new RuntimeException("Dim is not 128");
      for (int j = 0; j < 128; j++) {
        ret[i][j] = buf.getFloat();
      }
      VectorUtil.l2normalize(ret[i]);
    }
    return ret;
  }

  private int[][] readIvecs(InputStream is) throws Exception {
    byte[] bytes = is.readAllBytes();
    if (bytes.length == 0) return new int[0][0];
    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    int topK = buf.getInt();
    buf.position(0);
    int numVecs = bytes.length / (4 + topK * 4);
    int[][] ret = new int[numVecs][topK];
    for (int i = 0; i < numVecs; i++) {
      int dim = buf.getInt();
      if (dim != topK) throw new RuntimeException("Dim is not constant");
      for (int j = 0; j < topK; j++) {
        ret[i][j] = buf.getInt();
      }
    }
    return ret;
  }

  public void testIsoQuantPerformance() throws Exception {
    System.out.println("Warming up JIT & merging code paths... (JMH API equivalents)");
    // Thorough warmup
    for (int warmup = 0; warmup < 3; warmup++) {
        for (Config c : Config.values()) {
            measure(c, true);
        }
    }
    System.out.println("Warmup completed. Running rigorous measurements...");

    Result fp32 = measure(Config.BASELINE_FP32, false);
    Result int4 = measure(Config.BASELINE_INT4, false);
    Result int8 = measure(Config.BASELINE_INT8, false);
    Result iso4 = measure(Config.ISOQUANT_4BIT, false);
    Result iso8 = measure(Config.ISOQUANT_8BIT, false);
    Result iso4lz4 = measure(Config.ISOQUANT_4BIT_LZ4, false);
    Result iso8lz4 = measure(Config.ISOQUANT_8BIT_LZ4, false);

    System.out.println(
        "┌────────────────────────────────────────────────────────────────────────────────────────────────────┐");
    System.out.println(
        "│                      Recall, Speed & Storage Benchmark (d="
            + dimension
            + ", N="
            + numDocs
            + ")                      │");
    System.out.println(
        "├──────────────────────────┬────────────┬──────────────┬──────────────┬─────────────────┬─────────────┤");
    System.out.println(
        "│ Configuration            │   Recall   │ Index Time   │ Merge Time   │ Search Time     │ Storage Size│");
    System.out.println(
        "├──────────────────────────┼────────────┼──────────────┼──────────────┼─────────────────┼─────────────┤");
    printRow("Baseline (Raw FP32)", fp32);
    printRow("Baseline (INT4 SQ)", int4);
    printRow("Baseline (INT8 SQ)", int8);
    printRow("IsoQuant (4-bit LM)", iso4);
    printRow("IsoQuant (8-bit LM)", iso8);
    printRow("IsoQuant (4-bit LZ4)", iso4lz4);
    printRow("IsoQuant (8-bit LZ4)", iso8lz4);
    System.out.println(
        "└──────────────────────────┴────────────┴──────────────┴──────────────┴─────────────────┴─────────────┘");

    assertTrue("FP32 recall should be high", fp32.recall >= 0.90);
  }

  private void printRow(String name, Result res) {
    System.out.println(
        String.format(
            Locale.ROOT,
            "│ %-24s │   %.3f    │   %4d ms    │   %4d ms    │    %4d ms      │   %5d KB  │",
            name,
            res.recall,
            res.indexMs,
            res.mergeMs,
            res.searchMs,
            res.diskSizeBytes / 1024));
  }

  private Result measure(Config config, boolean isWarmup) throws Exception {
    KnnVectorsFormat format;
    if (config == Config.BASELINE_FP32) {
      format = new Lucene99HnswVectorsFormat(32, 200);
    } else if (config == Config.BASELINE_INT4) {
      format = new Lucene104HnswScalarQuantizedVectorsFormat(ScalarEncoding.PACKED_NIBBLE, 32, 200);
    } else if (config == Config.BASELINE_INT8) {
      format = new Lucene104HnswScalarQuantizedVectorsFormat(ScalarEncoding.UNSIGNED_BYTE, 32, 200);
    } else if (config == Config.ISOQUANT_4BIT) {
      format = new IsoQuantHnswVectorsFormat(32, 200, IsoEncoding.FOUR_BIT, IsoVariant.FAST, IsoCompress.NONE, 12345L);
    } else if (config == Config.ISOQUANT_8BIT) {
      format = new IsoQuantHnswVectorsFormat(32, 200, IsoEncoding.EIGHT_BIT, IsoVariant.FAST, IsoCompress.NONE, 12345L);
    } else if (config == Config.ISOQUANT_4BIT_LZ4) {
      format = new IsoQuantHnswVectorsFormat(32, 200, IsoEncoding.FOUR_BIT, IsoVariant.FAST, IsoCompress.LZ4, 12345L);
    } else {
      format = new IsoQuantHnswVectorsFormat(32, 200, IsoEncoding.EIGHT_BIT, IsoVariant.FAST, IsoCompress.LZ4, 12345L);
    }

    try (Directory dir = new ByteBuffersDirectory()) {
      IndexWriterConfig iwc =
          new IndexWriterConfig().setCodec(TestUtil.alwaysKnnVectorsFormat(format));
      
      // Control merges manually
      LogDocMergePolicy mergePolicy = new LogDocMergePolicy();
      mergePolicy.setMergeFactor(10); // arbitrary
      iwc.setMergePolicy(mergePolicy);
      iwc.setMergeScheduler(new SerialMergeScheduler());

      long indexTime = 0;
      long mergeTime = 0;

      try (IndexWriter writer = new IndexWriter(dir, iwc)) {
        long indexStart = System.nanoTime();
        
        int half = numDocs / 2;
        // Write first half
        for (int i = 0; i < half; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("field", dataset[i], VectorSimilarityFunction.COSINE));
          writer.addDocument(doc);
        }
        writer.flush(); // Force first segment

        // Write second half
        for (int i = half; i < numDocs; i++) {
          Document doc = new Document();
          doc.add(new KnnFloatVectorField("field", dataset[i], VectorSimilarityFunction.COSINE));
          writer.addDocument(doc);
        }
        writer.flush(); // Force second segment
        
        indexTime = (System.nanoTime() - indexStart) / 1_000_000;

        long mergeStart = System.nanoTime();
        writer.forceMerge(1);
        mergeTime = (System.nanoTime() - mergeStart) / 1_000_000;
        
        writer.commit();

        try (IndexReader reader = DirectoryReader.open(writer)) {
          IndexSearcher searcher = new IndexSearcher(reader);
          int totalCorrect = 0;

          long searchStart = System.nanoTime();
          for (int q = 0; q < numQueries; q++) {
            // Brute force ground truth
            Set<Integer> truth = new HashSet<>();
            for (int k = 0; k < TOP_K; k++) {
              truth.add(groundtruth[q][k]);
            }

            // KNN search
            TopDocs topDocs =
                searcher.search(new KnnFloatVectorQuery("field", queries[q], TOP_K), TOP_K);
            if (!isWarmup) {
                for (ScoreDoc sd : topDocs.scoreDocs) {
                  if (truth.contains(sd.doc)) totalCorrect++;
                }
            }
          }
          long searchTime = (System.nanoTime() - searchStart) / 1_000_000;

          long diskSize = 0;
          for (String file : dir.listAll()) {
            diskSize += dir.fileLength(file);
          }

          return new Result(
              (double) totalCorrect / (numQueries * TOP_K), indexTime, mergeTime, searchTime, diskSize);
        }
      }
    }
  }
}
