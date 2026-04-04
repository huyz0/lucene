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

package org.apache.lucene.codecs.isoquant;

import java.io.IOException;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.RotorQuant;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;

public class IsoQuantFlatVectorScorer implements FlatVectorsScorer {

  private final long seed;

  public IsoQuantFlatVectorScorer(long seed) {
    this.seed = seed;
  }

  @Override
  public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
      VectorSimilarityFunction similarityFunction,
      org.apache.lucene.index.KnnVectorValues vectorValues)
      throws IOException {
    if (!(vectorValues instanceof IsoQuantOffHeapVectorValues values)) {
      // During indexing/merging, the graph builder may pass exact FloatVectorValues.
      // We can just use exact float scoring for building the graph, which yields a better topology
      // anyway.
      return new RandomVectorScorerSupplier() {
        @Override
        public UpdateableRandomVectorScorer scorer() throws IOException {
          return new UpdateableRandomVectorScorer() {
            private int currentOrd = -1;

            @Override
            public void setScoringOrdinal(int node) {
              currentOrd = node;
            }

            @Override
            public float score(int node) throws IOException {
              float[] v1 =
                  ((org.apache.lucene.index.FloatVectorValues) vectorValues)
                      .vectorValue(currentOrd);
              float[] v2 =
                  ((org.apache.lucene.index.FloatVectorValues) vectorValues).vectorValue(node);
              return similarityFunction.compare(v1, v2);
            }

            @Override
            public int maxOrd() {
              return vectorValues.size();
            }
          };
        }

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
          return this;
        }
      };
    }

    int dim = values.dimension();
    int bits = values.getIsoEncoding().bits; 
    float[] centroids = RotorQuant.getLloydMaxCentroids(bits, dim);
    final float[] flatSymLut;
    if (bits == 4) {
      flatSymLut = new float[256]; // 16 * 16
      for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
          flatSymLut[i * 16 + j] = centroids[i] * centroids[j];
        }
      }
    } else {
      flatSymLut = null;
    }

    return new RandomVectorScorerSupplier() {
      @Override
      public UpdateableRandomVectorScorer scorer() throws IOException {
        int packedSliceLen = (dim * bits + 7) / 8;
        byte[] packed1 = new byte[packedSliceLen];
        float[] scaleAndNorm1 = new float[2];
        byte[] packed2 = new byte[packedSliceLen];
        float[] scaleAndNorm2 = new float[2];

        return new UpdateableRandomVectorScorer() {
          private int currentOrd = -1;

          @Override
          public void setScoringOrdinal(int node) throws IOException {
            currentOrd = node;
            values.getPackedBytesScaleAndNormSq(currentOrd, packed1, scaleAndNorm1);
          }

          @Override
          public float score(int node) throws IOException {
            if (node == currentOrd) return 1.0f; // Exact same node

            values.getPackedBytesScaleAndNormSq(node, packed2, scaleAndNorm2);
            
            float dotProduct = 0f;

            if (bits == 4) {
              for (int i = 0; i < dim; i += 2) {
                int b1 = packed1[i / 2] & 0xFF;
                int b2 = packed2[i / 2] & 0xFF;
                dotProduct += flatSymLut[((b1 >>> 4) << 4) + (b2 >>> 4)];
                if (i + 1 < dim) dotProduct += flatSymLut[((b1 & 0x0F) << 4) + (b2 & 0x0F)];
              }
            } else {
              for (int i = 0; i < dim; i++) {
                int b1 = packed1[i] & 0xFF;
                int b2 = packed2[i] & 0xFF;
                dotProduct += centroids[b1] * centroids[b2];
              }
            }

            dotProduct *= (scaleAndNorm1[0] * scaleAndNorm2[0]);

            if (similarityFunction == VectorSimilarityFunction.COSINE) {
              return Math.max((1.0f + dotProduct) / 2.0f, 0f);
            } else if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
              return Math.max(dotProduct, 0f);
            } else if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
              return org.apache.lucene.util.VectorUtil.scaleMaxInnerProductScore(dotProduct);
            } else if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
              float sqDistance = scaleAndNorm1[1] + scaleAndNorm2[1] - 2 * dotProduct;
              return Math.max(1.0f / (1.0f + sqDistance), 0f);
            }
            return dotProduct;
          }

          @Override
          public int maxOrd() {
            return values.size();
          }
        };
      }

      @Override
      public RandomVectorScorerSupplier copy() throws IOException {
        return this; // Should probably copy the vector values if multi-threaded builder
      }
    };
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(
      VectorSimilarityFunction similarityFunction,
      org.apache.lucene.index.KnnVectorValues vectorValues,
      float[] target)
      throws IOException {

    if (!(vectorValues instanceof IsoQuantOffHeapVectorValues values)) {
      // Fallback for exact tests or intermediate queries
      return new RandomVectorScorer() {
        @Override
        public float score(int node) throws IOException {
          float[] v = ((org.apache.lucene.index.FloatVectorValues) vectorValues).vectorValue(node);
          return similarityFunction.compare(target, v);
        }

        @Override
        public int maxOrd() {
          return vectorValues.size();
        }
      };
    }

    int dim = target.length;
    float[] codebook = RotorQuant.generateCodebook(seed, dim, values.getIsoVariant());

    // 1. Rotate the target (query) vector
    float[] rotatedQuery = target.clone();
    RotorQuant.rotate(rotatedQuery, codebook);

    int bits = values.getIsoEncoding().bits;

    // 2. We use the rotated query directly to avoid 64KB memory allocation and array caching issues
    float[] centroids = RotorQuant.getLloydMaxCentroids(bits, dim);

    // Pre-allocate buffers for scoring
    byte[] packedSlice = new byte[(dim * bits + 7) / 8];
    float[] scaleAndNormSlice = new float[2];
    
    float queryNormSq = org.apache.lucene.util.VectorUtil.dotProduct(target, target);

    return new RandomVectorScorer() {
      @Override
      public float score(int node) throws IOException {
        // Read packed indices, scale, and docNormSq
        values.getPackedBytesScaleAndNormSq(node, packedSlice, scaleAndNormSlice);

        // Compute hardware-accelerated dot product
        float dotProduct;
        if (bits == 4) {
          dotProduct = org.apache.lucene.util.VectorUtil.dotProductIsoQuant4Bit(packedSlice, rotatedQuery, centroids, dim);
        } else {
          dotProduct = org.apache.lucene.util.VectorUtil.dotProductIsoQuant8Bit(packedSlice, rotatedQuery, centroids, dim);
        }

        // Apply document norm correction scale
        dotProduct *= scaleAndNormSlice[0];

        if (similarityFunction == VectorSimilarityFunction.COSINE) {
          float score = Math.max((1.0f + dotProduct) / 2.0f, 0f);
          if (node == 0) System.out.println("DEBUG COSINE score node=0: " + score + ", dotProd=" + dotProduct + ", scale=" + scaleAndNormSlice[0]);
          return score;
        } else if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
          return Math.max(dotProduct, 0f);
        } else if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
          return org.apache.lucene.util.VectorUtil.scaleMaxInnerProductScore(dotProduct);
        } else if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
          float sqDistance = queryNormSq + scaleAndNormSlice[1] - 2 * dotProduct;
          if (node == 0) System.out.println("DEBUG EUCLIDEAN score node=0: " + (Math.max(1.0f / (1.0f + sqDistance), 0f)));
          return Math.max(1.0f / (1.0f + sqDistance), 0f);
        }
        return dotProduct;
      }

      @Override
      public int maxOrd() {
        return values.size();
      }
    };
  }

  @Override
  public RandomVectorScorer getRandomVectorScorer(
      VectorSimilarityFunction similarityFunction,
      org.apache.lucene.index.KnnVectorValues vectorValues,
      byte[] target)
      throws IOException {
    throw new UnsupportedOperationException("IsoQuant does not support byte target queries");
  }

  @Override
  public String toString() {
    return "IsoQuantFlatVectorScorer()";
  }
}
