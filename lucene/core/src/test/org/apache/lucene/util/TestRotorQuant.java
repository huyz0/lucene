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

package org.apache.lucene.util;

import java.util.Arrays;
import org.apache.lucene.codecs.isoquant.IsoVariant;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestRotorQuant extends LuceneTestCase {

  public void testNormPreservation() {
    int dims = 128;
    float[] codebook = RotorQuant.generateCodebook(42L, dims, IsoVariant.FAST);

    float[] vector = new float[dims];
    for (int i = 0; i < dims; i++) {
      vector[i] = random().nextFloat();
    }

    double originalNorm = Math.sqrt(VectorUtil.dotProduct(vector, vector));

    // Create copy and rotate
    float[] rotated = Arrays.copyOf(vector, vector.length);
    RotorQuant.rotate(rotated, codebook);

    double rotatedNorm = Math.sqrt(VectorUtil.dotProduct(rotated, rotated));

    assertEquals(originalNorm, rotatedNorm, 1e-4);
  }

  public void testDistancePreservation() {
    int dims = 256;
    float[] codebook = RotorQuant.generateCodebook(1337L, dims, IsoVariant.FAST);

    float[] v1 = new float[dims];
    float[] v2 = new float[dims];
    for (int i = 0; i < dims; i++) {
      v1[i] = random().nextFloat();
      v2[i] = random().nextFloat();
    }

    double originalDot = VectorUtil.dotProduct(v1, v2);

    float[] r1 = Arrays.copyOf(v1, v1.length);
    float[] r2 = Arrays.copyOf(v2, v2.length);

    RotorQuant.rotate(r1, codebook);
    RotorQuant.rotate(r2, codebook);

    double rotatedDot = VectorUtil.dotProduct(r1, r2);

    assertEquals(originalDot, rotatedDot, 1e-4);
  }

  public void testOutlierSpreading() {
    int dims = 64;
    float[] codebook = RotorQuant.generateCodebook(1L, dims, IsoVariant.FAST);

    float[] v1 = new float[dims];
    // Create an extreme outlier in the first 4-block
    v1[0] = 1000.0f;
    v1[1] = 0.5f;
    v1[2] = -0.5f;
    v1[3] = 0.0f;

    float[] r1 = Arrays.copyOf(v1, v1.length);
    RotorQuant.rotate(r1, codebook);

    // The energy (1000^2 + ...) should now be spread among r1[0], r1[1], r1[2], r1[3]
    // Due to the orthogonal properties of unit quaternions, none of the 4 dimensions
    // in the first block will have all the energy, yet the l2 norm of the block is preserved.
    float sumSq = r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2] + r1[3] * r1[3];
    float originalSumSq = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2] + v1[3] * v1[3];

    assertEquals(originalSumSq, sumSq, 1.0f);

    // Check that none of the values are exactly the original outlier
    assertTrue(r1[0] < 1000.0f - 1.0f || r1[0] > 1000.0f + 1.0f);
  }

  // ── Lloyd-Max Centroid Tests ─────────────────────────────────────────



  public void testLloydMaxCentroidsAreSymmetric() {
    for (int bits = 1; bits <= 5; bits++) {
      float[] centroids = RotorQuant.computeStandardNormalLloydMaxCentroids(bits);
      int n = centroids.length;

      // Centroids should be symmetric: c[i] = -c[n-1-i]
      for (int i = 0; i < n / 2; i++) {
        assertEquals(
            "Symmetry broken at bits=" + bits + " i=" + i,
            -centroids[i],
            centroids[n - 1 - i],
            1e-5f);
      }
    }
  }

  public void testLloydMaxCentroidsAreSorted() {
    for (int bits = 1; bits <= 6; bits++) {
      float[] centroids = RotorQuant.computeStandardNormalLloydMaxCentroids(bits);
      for (int i = 1; i < centroids.length; i++) {
        assertTrue(
            "Centroids not sorted at bits=" + bits + " i=" + i, centroids[i] > centroids[i - 1]);
      }
    }
  }

  public void testLloydMax4BitCentroids() {
    float[] centroids = RotorQuant.computeStandardNormalLloydMaxCentroids(4);
    assertEquals(16, centroids.length);

    // The outermost centroid for 16-level N(0,1) should be around ±2.73
    assertTrue("Outermost centroid too small: " + centroids[15], centroids[15] > 2.5f);
    assertTrue("Outermost centroid too large: " + centroids[15], centroids[15] < 3.0f);

    // The innermost centroid should be around ±0.128
    assertTrue("Innermost centroid too small: " + centroids[8], centroids[8] > 0.1f);
    assertTrue("Innermost centroid too large: " + centroids[8], centroids[8] < 0.2f);
  }

  public void testGetLloydMaxCentroidsScaling() {
    // For d=128: σ = 1/√128 ≈ 0.08839
    float[] c128 = RotorQuant.getLloydMaxCentroids(4, 128);
    // For d=256: σ = 1/√256 = 0.0625
    float[] c256 = RotorQuant.getLloydMaxCentroids(4, 256);

    // Centroids for d=256 should be smaller by factor √(128/256) = 1/√2
    float ratio = c128[7] / c256[7];
    assertEquals("Scaling ratio incorrect", Math.sqrt(256.0 / 128.0), ratio, 0.01);
  }

  public void testNearestCentroidIndex() {
    float[] centroids = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

    assertEquals(0, RotorQuant.nearestCentroidIndex(-5.0f, centroids));
    assertEquals(0, RotorQuant.nearestCentroidIndex(-1.6f, centroids));
    assertEquals(1, RotorQuant.nearestCentroidIndex(-1.0f, centroids));
    assertEquals(2, RotorQuant.nearestCentroidIndex(0.0f, centroids));
    assertEquals(2, RotorQuant.nearestCentroidIndex(0.3f, centroids));
    assertEquals(3, RotorQuant.nearestCentroidIndex(0.8f, centroids));
    assertEquals(4, RotorQuant.nearestCentroidIndex(5.0f, centroids));
  }

  public void testQuantizeVectorReturnsReconstructionNorm() {
    int dims = 128;
    float[] centroids = RotorQuant.getLloydMaxCentroids(4, dims);

    // Create a random unit vector
    float[] vector = new float[dims];
    for (int i = 0; i < dims; i++) {
      vector[i] = (float) (random().nextGaussian() / Math.sqrt(dims));
    }

    float reconNorm = RotorQuant.quantizeVector(vector, centroids);

    // Verify the returned norm matches the actual quantized vector norm
    float actualNormSq = 0f;
    for (float v : vector) {
      actualNormSq += v * v;
    }
    assertEquals(reconNorm, (float) Math.sqrt(actualNormSq), 1e-6f);

    // Verify all values are one of the centroids
    for (float v : vector) {
      boolean found = false;
      for (float c : centroids) {
        if (Math.abs(v - c) < 1e-7f) {
          found = true;
          break;
        }
      }
      assertTrue("Quantized value " + v + " not in centroids", found);
    }
  }

  public void testNormalCdfKnownValues() {
    // Φ(0) = 0.5
    assertEquals(0.5, RotorQuant.normalCdf(0.0), 1e-6);
    // Φ(1) ≈ 0.8413
    assertEquals(0.8413, RotorQuant.normalCdf(1.0), 1e-4);
    // Φ(-1) ≈ 0.1587
    assertEquals(0.1587, RotorQuant.normalCdf(-1.0), 1e-4);
    // Φ(2) ≈ 0.9772
    assertEquals(0.9772, RotorQuant.normalCdf(2.0), 1e-4);
  }
}
