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

import org.apache.lucene.codecs.isoquant.IsoVariant;

import java.util.Arrays;
import java.util.Random;

/**
 * Utility for fast block-wise 4D random quaternion rotations and Lloyd-Max optimal quantization.
 *
 * <p>After rotating a d-dimensional unit vector by a random orthogonal block rotation, each
 * coordinate follows the Beta distribution f(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2), which
 * for d ≥ 64 is well-approximated by N(0, 1/d). This known distribution allows pre-computation of
 * Lloyd-Max optimal quantization centroids — the information-theoretically optimal scalar quantizer
 * for this distribution.
 *
 * <p>The 4D block quaternion rotation transforms O(D²) dense rotations into O(D) and spreads energy
 * and outliers evenly across dimensions.
 *
 * @lucene.internal
 */
public final class RotorQuant {

  private RotorQuant() {}

  // ── Quaternion Rotation ──────────────────────────────────────────────

  /**
   * Generates a codebook of random 4D unit quaternions to rotate a vector. If {@code dimension} is
   * not a multiple of 4, the vector will be rotated in 4D blocks as much as possible, leaving
   * trailing elements unrotated.
   *
   * @param seed Random seed for reproducible codebook generation. Note the same seed is required
   *     for both indexing and search to ensure symmetric rotation.
   * @param dimension Dimensionality of vectors
   * @return A float array containing the unit quaternions (length = (dimension / 4) * 16)
   */
  public static float[] generateCodebook(long seed, int dimension, IsoVariant variant) {
    Random random = new Random(seed);
    int numBlocks = dimension / 4;
    float[] codebook = new float[numBlocks * 16];

    for (int i = 0; i < numBlocks; i++) {
      int offset = i * 16;
      if (variant == IsoVariant.PLANAR_2D) {
        double theta1 = random.nextDouble() * 2 * Math.PI;
        double theta2 = random.nextDouble() * 2 * Math.PI;
        float c1 = (float) Math.cos(theta1);
        float s1 = (float) Math.sin(theta1);
        float c2 = (float) Math.cos(theta2);
        float s2 = (float) Math.sin(theta2);

        codebook[offset] = c1; codebook[offset + 1] = s1; codebook[offset + 2] = 0f; codebook[offset + 3] = 0f;
        codebook[offset + 4] = -s1; codebook[offset + 5] = c1; codebook[offset + 6] = 0f; codebook[offset + 7] = 0f;
        codebook[offset + 8] = 0f; codebook[offset + 9] = 0f; codebook[offset + 10] = c2; codebook[offset + 11] = s2;
        codebook[offset + 12] = 0f; codebook[offset + 13] = 0f; codebook[offset + 14] = -s2; codebook[offset + 15] = c2;
      } else {
        float aL, bL, cL, dL;
        {
          double g1 = random.nextGaussian();
          double g2 = random.nextGaussian();
          double g3 = random.nextGaussian();
          double g4 = random.nextGaussian();
          double norm = Math.sqrt(g1 * g1 + g2 * g2 + g3 * g3 + g4 * g4);
          if (norm == 0) { aL = 1f; bL = 0f; cL = 0f; dL = 0f; } else {
            aL = (float) (g1 / norm); bL = (float) (g2 / norm); cL = (float) (g3 / norm); dL = (float) (g4 / norm);
          }
        }

        float[] lMat = new float[] {
           aL, -bL, -cL, -dL,
           bL,  aL, -dL,  cL,
           cL,  dL,  aL, -bL,
           dL, -cL,  bL,  aL
        };

        if (variant == IsoVariant.FAST) {
          for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
               codebook[offset + c * 4 + r] = lMat[r * 4 + c];
            }
          }
        } else {
          float aR, bR, cR, dR;
          {
            double g1 = random.nextGaussian();
            double g2 = random.nextGaussian();
            double g3 = random.nextGaussian();
            double g4 = random.nextGaussian();
            double norm = Math.sqrt(g1 * g1 + g2 * g2 + g3 * g3 + g4 * g4);
            if (norm == 0) { aR = 1f; bR = 0f; cR = 0f; dR = 0f; } else {
              aR = (float) (g1 / norm); bR = (float) (g2 / norm); cR = (float) (g3 / norm); dR = (float) (g4 / norm);
            }
          }

          float[] rMat = new float[] {
             aR,  bR,  cR,  dR,
            -bR,  aR, -dR,  cR,
            -cR,  dR,  aR, -bR,
            -dR, -cR,  bR,  aR
          };

          for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
              float sum = 0;
              for (int k = 0; k < 4; k++) {
                sum += lMat[r * 4 + k] * rMat[k * 4 + c];
              }
              codebook[offset + c * 4 + r] = sum;
            }
          }
        }
      }
    }
    return codebook;
  }

  /**
   * Rotates a dense float vector using the provided 4D unit quaternions. The matrix transformation
   * guarantees properties of orthogonal rotation (preserving vector norm). Modifies the vector
   * in-place.
   *
   * @param vector the vector to rotate
   * @param codebook the list of unit quaternions generated by {@link #generateCodebook}
   */
  public static void rotate(float[] vector, float[] codebook) {
    VectorUtil.rotorQuantRotate(vector, codebook);
  }

  /**
   * Generates the transposed (inverse) rotation matrix codebook given an existing codebook. This
   * simply negates the non-scalar parts of the underlying quaternions (b, c, d).
   *
   * @param codebook the list of unit quaternions generated by {@link #generateCodebook}
   * @return the inverse codebook
   */
  public static float[] generateInverseCodebook(float[] codebook) {
    float[] inverse = new float[codebook.length];
    for (int i = 0; i < codebook.length; i += 16) {
      float a = codebook[i];
      float b = codebook[i + 1];
      float c = codebook[i + 2];
      float d = codebook[i + 3];

      // The conjugate of (a, b, c, d) is (a, -b, -c, -d)
      // column 1 (multiplier for x)
      inverse[i] = a;
      inverse[i + 1] = -b;
      inverse[i + 2] = -c;
      inverse[i + 3] = -d;
      // column 2 (multiplier for y)
      inverse[i + 4] = b;
      inverse[i + 5] = a;
      inverse[i + 6] = -d;
      inverse[i + 7] = c;
      // column 3 (multiplier for z)
      inverse[i + 8] = c;
      inverse[i + 9] = d;
      inverse[i + 10] = a;
      inverse[i + 11] = -b;
      // column 4 (multiplier for w)
      inverse[i + 12] = d;
      inverse[i + 13] = -c;
      inverse[i + 14] = b;
      inverse[i + 15] = a;
    }
    return inverse;
  }

  // ── Lloyd-Max Optimal Quantization ──────────────────────────────────




  /**
   * Pre-computed Lloyd-Max optimal centroids for the standard normal distribution N(0, 1). To get
   * centroids for dimension d, multiply each value by {@code 1.0 / sqrt(d)}.
   *
   * <p>These match the centroids used in llama.cpp's iso3 quantization (ggml-iso-quant.c
   * ISO_CENTROIDS_3BIT) when scaled for d=128.
   */
  static final float[] STANDARD_NORMAL_LLOYD_MAX_3BIT = {
    -2.15224f, -1.34412f, -0.75633f, -0.24509f, 0.24509f, 0.75633f, 1.34412f, 2.15224f,
  };

  /** Pre-computed Lloyd-Max optimal centroids for N(0, 1) at 4-bit (16 levels). */
  static final float[] STANDARD_NORMAL_LLOYD_MAX_4BIT = {
    -2.73319f, -2.06906f, -1.61816f, -1.25605f, -0.94218f, -0.65676f, -0.38826f, -0.12839f,
    0.12839f, 0.38826f, 0.65676f, 0.94218f, 1.25605f, 1.61816f, 2.06906f, 2.73319f,
  };

  /** Pre-computed Lloyd-Max optimal centroids for N(0, 1) at 8-bit (256 levels). Generated via computeStandardNormalLloydMaxCentroids(8). */
  static final float[] STANDARD_NORMAL_LLOYD_MAX_8BIT = computeStandardNormalLloydMaxCentroids(8);

  /** Standard normal probability density function φ(z) = (1/√(2π)) · exp(-z²/2). */
  static double normalPdf(double z) {
    return 0.3989422804014327 * Math.exp(-0.5 * z * z);
  }

  /**
   * Standard normal cumulative distribution function Φ(z). Uses the Abramowitz & Stegun rational
   * approximation (max error &lt; 7.5e-8).
   */
  static double normalCdf(double z) {
    if (z < -8.0) return 0.0;
    if (z > 8.0) return 1.0;
    double t = 1.0 / (1.0 + 0.2316419 * Math.abs(z));
    double prob =
        normalPdf(z)
            * t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return z > 0 ? 1.0 - prob : prob;
  }

  /**
   * Computes Lloyd-Max optimal quantization centroids for the standard normal distribution N(0, 1).
   *
   * <p>The algorithm iteratively solves the Lloyd-Max conditions using the closed-form conditional
   * mean of the Gaussian distribution:
   *
   * <pre>
   *   E[X | a &lt; X &lt; b] = σ · [φ(a/σ) - φ(b/σ)] / [Φ(b/σ) - Φ(a/σ)]
   * </pre>
   *
   * where φ is the standard normal PDF and Φ is the standard normal CDF.
   *
   * @param bits number of quantization bits (1-8)
   * @return sorted array of 2^bits optimal centroids for N(0, 1)
   */
  public static float[] computeStandardNormalLloydMaxCentroids(int bits) {
    if (bits < 1 || bits > 8) {
      throw new IllegalArgumentException("bits must be between 1 and 8, got: " + bits);
    }
    int nLevels = 1 << bits;
    double lo = -3.5;
    double hi = 3.5;

    // Initialize centroids uniformly in [-3.5, 3.5]
    double[] centroids = new double[nLevels];
    for (int i = 0; i < nLevels; i++) {
      centroids[i] = lo + (hi - lo) * (i + 0.5) / nLevels;
    }

    // Lloyd-Max iteration using closed-form Gaussian conditional mean
    for (int iter = 0; iter < 200; iter++) {
      // Compute boundaries (midpoints between adjacent centroids)
      double[] edges = new double[nLevels + 1];
      edges[0] = -10.0; // effectively -∞
      edges[nLevels] = 10.0; // effectively +∞
      for (int i = 0; i < nLevels - 1; i++) {
        edges[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0;
      }

      // Update centroids as conditional expectations E[X | edge_i < X < edge_{i+1}]
      double maxShift = 0;
      for (int i = 0; i < nLevels; i++) {
        double a = edges[i];
        double b = edges[i + 1];

        // For N(0,1): E[X | a<X<b] = [φ(a) - φ(b)] / [Φ(b) - Φ(a)]
        double denominator = normalCdf(b) - normalCdf(a);
        if (denominator > 1e-15) {
          double numerator = normalPdf(a) - normalPdf(b);
          double newCentroid = numerator / denominator;
          maxShift = Math.max(maxShift, Math.abs(newCentroid - centroids[i]));
          centroids[i] = newCentroid;
        }
      }

      if (maxShift < 1e-10) break;
    }

    float[] result = new float[nLevels];
    for (int i = 0; i < nLevels; i++) {
      result[i] = (float) centroids[i];
    }
    return result;
  }

  /**
   * Returns Lloyd-Max optimal centroids for quantizing coordinates of a rotated d-dimensional unit
   * vector. After rotation, coordinates are approximately N(0, 1/d), so centroids are the standard
   * normal centroids scaled by 1/√d.
   *
   * <p>Uses pre-computed tables for 3-bit and 4-bit; computes on-the-fly for other bit widths.
   *
   * @param dimension vector dimensionality (determines the coordinate distribution variance)
   * @return sorted array of 16 optimal centroids (4-bit)
   */
  public static float[] getLloydMaxCentroids(int bits, int dimension) {
    float[] standardCentroids;
    if (bits == 4) {
      standardCentroids = STANDARD_NORMAL_LLOYD_MAX_4BIT;
    } else if (bits == 8) {
      standardCentroids = STANDARD_NORMAL_LLOYD_MAX_8BIT;
    } else {
      throw new IllegalArgumentException("Only 4-bit and 8-bit are supported");
    }

    // Scale by σ = 1/√d for the post-rotation coordinate distribution
    float sigma = (float) (1.0 / Math.sqrt(dimension));
    float[] scaled = new float[standardCentroids.length];
    for (int i = 0; i < standardCentroids.length; i++) {
      scaled[i] = standardCentroids[i] * sigma;
    }
    return scaled;
  }

  /**
   * Finds the index of the nearest centroid to the given value using binary search. Centroids must
   * be sorted in ascending order.
   *
   * @param value the value to quantize
   * @param centroids sorted array of centroid values
   * @return index of the nearest centroid
   */
  public static int nearestCentroidIndex(float value, float[] centroids) {
    int idx = Arrays.binarySearch(centroids, value);
    if (idx >= 0) {
      return idx; // exact match
    }
    // binarySearch returns -(insertion point) - 1
    int insertionPoint = -(idx + 1);
    if (insertionPoint == 0) {
      return 0;
    }
    if (insertionPoint >= centroids.length) {
      return centroids.length - 1;
    }
    // Compare distances to the two neighbors
    float distLow = Math.abs(value - centroids[insertionPoint - 1]);
    float distHigh = Math.abs(value - centroids[insertionPoint]);
    return distLow <= distHigh ? insertionPoint - 1 : insertionPoint;
  }

  /**
   * Quantizes a single value to its nearest Lloyd-Max centroid value.
   *
   * @param value the value to quantize
   * @param centroids sorted array of centroid values
   * @return the centroid value nearest to the input
   */
  public static float quantizeToNearestCentroid(float value, float[] centroids) {
    return centroids[nearestCentroidIndex(value, centroids)];
  }

  /**
   * Quantizes an entire vector in-place using Lloyd-Max centroids, and returns the L2 norm of the
   * resulting quantized vector (for norm correction).
   *
   * <p>This implements the core quantization step from the RotorQuant/IsoQuant pipeline: each
   * coordinate of a rotated unit vector is snapped to its nearest optimal centroid.
   *
   * @param vector the rotated unit vector to quantize (modified in-place)
   * @param centroids sorted Lloyd-Max centroid values for the coordinate distribution
   * @return L2 norm of the quantized vector (for computing corrected norm)
   */
  public static float quantizeVector(float[] vector, float[] centroids) {
    if (centroids.length == 16) {
        float normSq = 0f;
        float b0 = (centroids[0] + centroids[1]) * 0.5f;
        float b1 = (centroids[1] + centroids[2]) * 0.5f;
        float b2 = (centroids[2] + centroids[3]) * 0.5f;
        float b3 = (centroids[3] + centroids[4]) * 0.5f;
        float b4 = (centroids[4] + centroids[5]) * 0.5f;
        float b5 = (centroids[5] + centroids[6]) * 0.5f;
        float b6 = (centroids[6] + centroids[7]) * 0.5f;
        float b7 = (centroids[7] + centroids[8]) * 0.5f;
        float b8 = (centroids[8] + centroids[9]) * 0.5f;
        float b9 = (centroids[9] + centroids[10]) * 0.5f;
        float b10 = (centroids[10] + centroids[11]) * 0.5f;
        float b11 = (centroids[11] + centroids[12]) * 0.5f;
        float b12 = (centroids[12] + centroids[13]) * 0.5f;
        float b13 = (centroids[13] + centroids[14]) * 0.5f;
        float b14 = (centroids[14] + centroids[15]) * 0.5f;
        for (int i = 0; i < vector.length; i++) {
            float val = vector[i];
            int idx = 0;
            idx += (val >= b0) ? 1 : 0; idx += (val >= b1) ? 1 : 0; idx += (val >= b2) ? 1 : 0;
            idx += (val >= b3) ? 1 : 0; idx += (val >= b4) ? 1 : 0; idx += (val >= b5) ? 1 : 0;
            idx += (val >= b6) ? 1 : 0; idx += (val >= b7) ? 1 : 0; idx += (val >= b8) ? 1 : 0;
            idx += (val >= b9) ? 1 : 0; idx += (val >= b10) ? 1 : 0; idx += (val >= b11) ? 1 : 0;
            idx += (val >= b12) ? 1 : 0; idx += (val >= b13) ? 1 : 0; idx += (val >= b14) ? 1 : 0;
            vector[i] = centroids[idx];
            normSq += vector[i] * vector[i];
        }
        return (float) Math.sqrt(normSq);
    }
    float normSq = 0f;
    for (int i = 0; i < vector.length; i++) {
      vector[i] = quantizeToNearestCentroid(vector[i], centroids);
      normSq += vector[i] * vector[i];
    }
    return (float) Math.sqrt(normSq);
  }

  /**
   * Quantizes an entire vector to n-bit, returning a packed byte[] configuration. Format: byte[dim
   * * bits / 8].
   *
   * @param vector the rotated unit vector to quantize
   * @param centroids sorted Lloyd-Max centroid values
   * @param bits the bit width (1, 2, 4, 8)
   * @param normOut array of length 1 to receive the L2 norm of the reconstructed vector
   * @return byte[] containing the packed values
   */
  public static byte[] quantize(float[] vector, float[] centroids, int bits, float[] normOut) {
    if ((1 << bits) != centroids.length) {
      throw new IllegalArgumentException("bits configuration does not match number of centroids");
    }
    float normSq = 0f;
    byte[] packed = new byte[vector.length * bits / 8];
    int valuesPerByte = 8 / bits;

    if (bits == 4) {
      float b0 = (centroids[0] + centroids[1]) * 0.5f;
      float b1 = (centroids[1] + centroids[2]) * 0.5f;
      float b2 = (centroids[2] + centroids[3]) * 0.5f;
      float b3 = (centroids[3] + centroids[4]) * 0.5f;
      float b4 = (centroids[4] + centroids[5]) * 0.5f;
      float b5 = (centroids[5] + centroids[6]) * 0.5f;
      float b6 = (centroids[6] + centroids[7]) * 0.5f;
      float b7 = (centroids[7] + centroids[8]) * 0.5f;
      float b8 = (centroids[8] + centroids[9]) * 0.5f;
      float b9 = (centroids[9] + centroids[10]) * 0.5f;
      float b10 = (centroids[10] + centroids[11]) * 0.5f;
      float b11 = (centroids[11] + centroids[12]) * 0.5f;
      float b12 = (centroids[12] + centroids[13]) * 0.5f;
      float b13 = (centroids[13] + centroids[14]) * 0.5f;
      float b14 = (centroids[14] + centroids[15]) * 0.5f;

      for (int i = 0; i < vector.length; i += 2) {
        float val0 = vector[i];
        int idx0 = 0;
        idx0 += (val0 >= b0) ? 1 : 0; idx0 += (val0 >= b1) ? 1 : 0; idx0 += (val0 >= b2) ? 1 : 0;
        idx0 += (val0 >= b3) ? 1 : 0; idx0 += (val0 >= b4) ? 1 : 0; idx0 += (val0 >= b5) ? 1 : 0;
        idx0 += (val0 >= b6) ? 1 : 0; idx0 += (val0 >= b7) ? 1 : 0; idx0 += (val0 >= b8) ? 1 : 0;
        idx0 += (val0 >= b9) ? 1 : 0; idx0 += (val0 >= b10) ? 1 : 0; idx0 += (val0 >= b11) ? 1 : 0;
        idx0 += (val0 >= b12) ? 1 : 0; idx0 += (val0 >= b13) ? 1 : 0; idx0 += (val0 >= b14) ? 1 : 0;

        float val1 = (i + 1 < vector.length) ? vector[i + 1] : 0f;
        int idx1 = 0;
        idx1 += (val1 >= b0) ? 1 : 0; idx1 += (val1 >= b1) ? 1 : 0; idx1 += (val1 >= b2) ? 1 : 0;
        idx1 += (val1 >= b3) ? 1 : 0; idx1 += (val1 >= b4) ? 1 : 0; idx1 += (val1 >= b5) ? 1 : 0;
        idx1 += (val1 >= b6) ? 1 : 0; idx1 += (val1 >= b7) ? 1 : 0; idx1 += (val1 >= b8) ?  1 : 0;
        idx1 += (val1 >= b9) ? 1 : 0; idx1 += (val1 >= b10) ? 1 : 0; idx1 += (val1 >= b11) ? 1 : 0;
        idx1 += (val1 >= b12) ? 1 : 0; idx1 += (val1 >= b13) ? 1 : 0; idx1 += (val1 >= b14) ? 1 : 0;

        normSq += centroids[idx0] * centroids[idx0];
        if (i + 1 < vector.length) {
            normSq += centroids[idx1] * centroids[idx1];
        }
        packed[i / 2] = (byte) ((idx0 << 4) | idx1);
      }
    } else {
        for (int i = 0; i < vector.length; i += valuesPerByte) {
          byte b = 0;
          for (int v = 0; v < valuesPerByte; v++) {
            if (i + v < vector.length) {
              int idx = nearestCentroidIndex(vector[i + v], centroids);
              normSq += centroids[idx] * centroids[idx];
              int shift = (valuesPerByte - 1 - v) * bits;
              b |= (idx << shift);
            }
          }
          packed[i / valuesPerByte] = b;
        }
    }

    if (normOut != null && normOut.length > 0) {
      normOut[0] = (float) Math.sqrt(normSq);
    }
    return packed;
  }
}
