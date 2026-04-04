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

package org.apache.lucene.internal.vectorization;

import static org.apache.lucene.util.VectorUtil.isUnitVector;

import org.apache.lucene.util.BitUtil;
import org.apache.lucene.util.Constants;
import org.apache.lucene.util.SuppressForbidden;

final class DefaultVectorUtilSupport implements VectorUtilSupport {

  DefaultVectorUtilSupport() {}

  // the way FMA should work! if available use it, otherwise fall back to mul/add
  @SuppressForbidden(reason = "Uses FMA only where fast and carefully contained")
  private static float fma(float a, float b, float c) {
    if (Constants.HAS_FAST_SCALAR_FMA) {
      return Math.fma(a, b, c);
    } else {
      return a * b + c;
    }
  }

  @Override
  public float dotProduct(float[] a, float[] b) {
    float res = 0f;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float acc1 = 0;
      float acc2 = 0;
      float acc3 = 0;
      float acc4 = 0;
      int upperBound = a.length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        acc1 = fma(a[i], b[i], acc1);
        acc2 = fma(a[i + 1], b[i + 1], acc2);
        acc3 = fma(a[i + 2], b[i + 2], acc3);
        acc4 = fma(a[i + 3], b[i + 3], acc4);
      }
      res += acc1 + acc2 + acc3 + acc4;
    }

    for (; i < a.length; i++) {
      res = fma(a[i], b[i], res);
    }
    return res;
  }

  @Override
  public float cosine(float[] a, float[] b) {
    float sum = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float sum1 = 0;
      float sum2 = 0;
      float norm1_1 = 0;
      float norm1_2 = 0;
      float norm2_1 = 0;
      float norm2_2 = 0;

      int upperBound = a.length & ~(2 - 1);
      for (; i < upperBound; i += 2) {
        // one
        sum1 = fma(a[i], b[i], sum1);
        norm1_1 = fma(a[i], a[i], norm1_1);
        norm2_1 = fma(b[i], b[i], norm2_1);

        // two
        sum2 = fma(a[i + 1], b[i + 1], sum2);
        norm1_2 = fma(a[i + 1], a[i + 1], norm1_2);
        norm2_2 = fma(b[i + 1], b[i + 1], norm2_2);
      }
      sum += sum1 + sum2;
      norm1 += norm1_1 + norm1_2;
      norm2 += norm2_1 + norm2_2;
    }

    for (; i < a.length; i++) {
      sum = fma(a[i], b[i], sum);
      norm1 = fma(a[i], a[i], norm1);
      norm2 = fma(b[i], b[i], norm2);
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  @Override
  public float squareDistance(float[] a, float[] b) {
    float res = 0;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float acc1 = 0;
      float acc2 = 0;
      float acc3 = 0;
      float acc4 = 0;

      int upperBound = a.length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        // one
        float diff1 = a[i] - b[i];
        acc1 = fma(diff1, diff1, acc1);

        // two
        float diff2 = a[i + 1] - b[i + 1];
        acc2 = fma(diff2, diff2, acc2);

        // three
        float diff3 = a[i + 2] - b[i + 2];
        acc3 = fma(diff3, diff3, acc3);

        // four
        float diff4 = a[i + 3] - b[i + 3];
        acc4 = fma(diff4, diff4, acc4);
      }
      res += acc1 + acc2 + acc3 + acc4;
    }

    for (; i < a.length; i++) {
      float diff = a[i] - b[i];
      res = fma(diff, diff, res);
    }
    return res;
  }

  @Override
  public int dotProduct(byte[] a, byte[] b) {
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      total += a[i] * b[i];
    }
    return total;
  }

  @Override
  public int uint8DotProduct(byte[] a, byte[] b) {
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      total += Byte.toUnsignedInt(a[i]) * Byte.toUnsignedInt(b[i]);
    }
    return total;
  }

  @Override
  public int int4DotProduct(byte[] a, byte[] b) {
    return dotProduct(a, b);
  }

  @Override
  public int int4DotProductSinglePacked(byte[] unpacked, byte[] packed) {
    int total = 0;
    for (int i = 0; i < packed.length; i++) {
      byte packedByte = packed[i];
      byte unpacked1 = unpacked[i];
      byte unpacked2 = unpacked[i + packed.length];
      total += (packedByte & 0x0F) * unpacked2;
      total += ((packedByte & 0xFF) >> 4) * unpacked1;
    }
    return total;
  }

  @Override
  public int int4DotProductBothPacked(byte[] a, byte[] b) {
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      byte aByte = a[i];
      byte bByte = b[i];
      total += (aByte & 0x0F) * (bByte & 0x0F);
      total += ((aByte & 0xFF) >> 4) * ((bByte & 0xFF) >> 4);
    }
    return total;
  }

  @Override
  public float cosine(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int sum = 0;
    int norm1 = 0;
    int norm2 = 0;

    for (int i = 0; i < a.length; i++) {
      byte elem1 = a[i];
      byte elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  @Override
  public int squareDistance(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int squareSum = 0;
    for (int i = 0; i < a.length; i++) {
      int diff = a[i] - b[i];
      squareSum += diff * diff;
    }
    return squareSum;
  }

  @Override
  public int int4SquareDistance(byte[] a, byte[] b) {
    return squareDistance(a, b);
  }

  @Override
  public int int4SquareDistanceSinglePacked(byte[] unpacked, byte[] packed) {
    int total = 0;
    for (int i = 0; i < packed.length; i++) {
      byte packedByte = packed[i];
      byte unpacked1 = unpacked[i];
      byte unpacked2 = unpacked[i + packed.length];

      int diff1 = (packedByte & 0x0F) - unpacked2;
      int diff2 = ((packedByte & 0xFF) >> 4) - unpacked1;

      total += diff1 * diff1 + diff2 * diff2;
    }
    return total;
  }

  @Override
  public int int4SquareDistanceBothPacked(byte[] a, byte[] b) {
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      byte aByte = a[i];
      byte bByte = b[i];

      int diff1 = (aByte & 0x0F) - (bByte & 0x0F);
      int diff2 = ((aByte & 0xFF) >> 4) - ((bByte & 0xFF) >> 4);

      total += diff1 * diff1 + diff2 * diff2;
    }
    return total;
  }

  @Override
  public int uint8SquareDistance(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^16, since max(ubyte * ubyte) = 2^16.
    int squareSum = 0;
    for (int i = 0; i < a.length; i++) {
      int diff = Byte.toUnsignedInt(a[i]) - Byte.toUnsignedInt(b[i]);
      squareSum += diff * diff;
    }
    return squareSum;
  }

  @Override
  public int findNextGEQ(int[] buffer, int target, int from, int to) {
    for (int i = from; i < to; ++i) {
      if (buffer[i] >= target) {
        return i;
      }
    }
    return to;
  }

  @Override
  public long int4BitDotProduct(byte[] int4Quantized, byte[] binaryQuantized) {
    return int4BitDotProductImpl(int4Quantized, binaryQuantized);
  }

  public static long int4BitDotProductImpl(byte[] q, byte[] d) {
    assert q.length == d.length * 4;
    return int4BitDotProductImpl(q, d, 0, d.length);
  }

  @Override
  public long int4DibitDotProduct(byte[] int4Quantized, byte[] dibitQuantized) {
    return int4DibitDotProductImpl(int4Quantized, dibitQuantized);
  }

  /**
   * Computes the dot product between a transposed 4-bit query vector and a transposed 2-bit
   * document vector. The dibit vector has 2 stripes (lower bits first, then upper bits), so the
   * scoring is two passes of int4-bit dot product with results shifted appropriately.
   *
   * @param q transposed 4-bit query vector (4 stripes, each of size q.length/4)
   * @param d transposed 2-bit document vector (2 stripes, each of size d.length/2)
   * @return the dot product
   */
  public static long int4DibitDotProductImpl(byte[] q, byte[] d) {
    assert q.length == d.length * 2;
    int stripeSize = d.length / 2;
    long ret0 = int4BitDotProductImpl(q, d, 0, stripeSize);
    long ret1 = int4BitDotProductImpl(q, d, stripeSize, stripeSize);

    return ret0 + (ret1 << 1);
  }

  /**
   * Helper method to compute int4-bit dot product with a specific stripe of the document vector.
   *
   * @param q transposed 4-bit query vector (4 stripes)
   * @param d transposed document vector
   * @param dOffset offset into d for the stripe to use
   * @param stripeSize size of each stripe
   * @return the dot product for this stripe
   */
  private static long int4BitDotProductImpl(byte[] q, byte[] d, int dOffset, int stripeSize) {
    long ret = 0;
    for (int i = 0; i < 4; i++) {
      int r = 0;
      long subRet = 0;
      for (final int upperBound = stripeSize & -Integer.BYTES; r < upperBound; r += Integer.BYTES) {
        subRet +=
            Integer.bitCount(
                (int) BitUtil.VH_NATIVE_INT.get(q, i * stripeSize + r)
                    & (int) BitUtil.VH_NATIVE_INT.get(d, dOffset + r));
      }
      for (; r < stripeSize; r++) {
        subRet += Integer.bitCount((q[i * stripeSize + r] & d[dOffset + r]) & 0xFF);
      }
      ret += subRet << i;
    }
    return ret;
  }

  @Override
  public float minMaxScalarQuantize(
      float[] vector, byte[] dest, float scale, float alpha, float minQuantile, float maxQuantile) {
    return new ScalarQuantizer(alpha, scale, minQuantile, maxQuantile).quantize(vector, dest, 0);
  }

  @Override
  public float recalculateScalarQuantizationOffset(
      byte[] vector,
      float oldAlpha,
      float oldMinQuantile,
      float scale,
      float alpha,
      float minQuantile,
      float maxQuantile) {
    return new ScalarQuantizer(alpha, scale, minQuantile, maxQuantile)
        .recalculateOffset(vector, 0, oldAlpha, oldMinQuantile);
  }

  static class ScalarQuantizer {
    private final float alpha;
    private final float scale;
    private final float minQuantile, maxQuantile;

    ScalarQuantizer(float alpha, float scale, float minQuantile, float maxQuantile) {
      this.alpha = alpha;
      this.scale = scale;
      this.minQuantile = minQuantile;
      this.maxQuantile = maxQuantile;
    }

    float quantize(float[] vector, byte[] dest, int start) {
      assert vector.length == dest.length;
      float correction = 0;
      for (int i = start; i < vector.length; i++) {
        correction += quantizeFloat(vector[i], dest, i);
      }
      return correction;
    }

    float recalculateOffset(byte[] vector, int start, float oldAlpha, float oldMinQuantile) {
      float correction = 0;
      for (int i = start; i < vector.length; i++) {
        // undo the old quantization
        float v = (oldAlpha * Byte.toUnsignedInt(vector[i])) + oldMinQuantile;
        correction += quantizeFloat(v, null, 0);
      }
      return correction;
    }

    private float quantizeFloat(float v, byte[] dest, int destIndex) {
      assert dest == null || destIndex < dest.length;
      // Make sure the value is within the quantile range, cutting off the tails
      // see first parenthesis in equation: byte = (float - minQuantile) * 127/(maxQuantile -
      // minQuantile)
      float dx = v - minQuantile;
      float dxc = Math.max(minQuantile, Math.min(maxQuantile, v)) - minQuantile;
      // Scale the value to the range [0, 127], this is our quantized value
      // scale = 127/(maxQuantile - minQuantile)
      int roundedDxs = Math.round(scale * dxc);
      // We multiply by `alpha` here to get the quantized value back into the original range
      // to aid in calculating the corrective offset
      float dxq = roundedDxs * alpha;
      if (dest != null) {
        dest[destIndex] = (byte) roundedDxs;
      }
      // Calculate the corrective offset that needs to be applied to the score
      // in addition to the `byte * minQuantile * alpha` term in the equation
      // we add the `(dx - dxq) * dxq` term to account for the fact that the quantized value
      // will be rounded to the nearest whole number and lose some accuracy
      // Additionally, we account for the global correction of `minQuantile^2` in the equation
      return minQuantile * (v - minQuantile / 2.0F) + (dx - dxq) * dxq;
    }
  }

  @Override
  public int filterByScore(
      int[] docBuffer, double[] scoreBuffer, double minScoreInclusive, int upTo) {
    int newSize = 0;
    for (int i = 0; i < upTo; ++i) {
      int doc = docBuffer[i];
      double score = scoreBuffer[i];
      docBuffer[newSize] = doc;
      scoreBuffer[newSize] = score;
      if (score >= minScoreInclusive) {
        newSize++;
      }
    }
    return newSize;
  }

  @Override
  public float[] l2normalize(float[] v, boolean throwOnZero) {
    double squaredNorm = this.dotProduct(v, v);
    if (squaredNorm == 0) {
      if (throwOnZero) {
        throw new IllegalArgumentException("Cannot normalize a zero-length vector");
      } else {
        return v;
      }
    }
    if (isUnitVector(squaredNorm)) {
      return v;
    }

    int dim = v.length;
    double l2norm = Math.sqrt(squaredNorm);
    for (int i = 0; i < dim; i++) {
      v[i] /= (float) l2norm;
    }
    return v;
  }

  @Override
  public void expand8(int[] arr) {
    // BLOCK_SIZE is 256
    for (int i = 0; i < 64; ++i) {
      int l = arr[i];
      arr[i] = (l >>> 24) & 0xFF;
      arr[64 + i] = (l >>> 16) & 0xFF;
      arr[128 + i] = (l >>> 8) & 0xFF;
      arr[192 + i] = l & 0xFF;
    }
  }

  @Override
  public void rotorQuantRotate(float[] vector, float[] codebook) {
    int blocks = vector.length / 4;
    for (int b = 0; b < blocks; b++) {
      int vOff = b * 4;
      int cOff = b * 16;
      float x = vector[vOff];
      float y = vector[vOff + 1];
      float z = vector[vOff + 2];
      float w = vector[vOff + 3];

      vector[vOff] =
          fma(
              codebook[cOff],
              x,
              fma(codebook[cOff + 4], y, fma(codebook[cOff + 8], z, codebook[cOff + 12] * w)));
      vector[vOff + 1] =
          fma(
              codebook[cOff + 1],
              x,
              fma(codebook[cOff + 5], y, fma(codebook[cOff + 9], z, codebook[cOff + 13] * w)));
      vector[vOff + 2] =
          fma(
              codebook[cOff + 2],
              x,
              fma(codebook[cOff + 6], y, fma(codebook[cOff + 10], z, codebook[cOff + 14] * w)));
      vector[vOff + 3] =
          fma(
              codebook[cOff + 3],
              x,
              fma(codebook[cOff + 7], y, fma(codebook[cOff + 11], z, codebook[cOff + 15] * w)));
    }
  }

  @Override
  public float dotProductIsoQuant4Bit(byte[] packed, float[] query, float[] centroids, int dim) {
    float d0 = 0f, d1 = 0f, d2 = 0f, d3 = 0f;
    int i = 0;
    
    // Process 8 dimensions per loop iteration (4 bytes)
    int limit = dim - (dim % 8);
    for (; i < limit; i += 8) {
        int b0 = packed[i >> 1] & 0xFF;
        int b1 = packed[(i >> 1) + 1] & 0xFF;
        int b2 = packed[(i >> 1) + 2] & 0xFF;
        int b3 = packed[(i >> 1) + 3] & 0xFF;

        d0 += query[i] * centroids[b0 >>> 4] + query[i + 1] * centroids[b0 & 0x0F];
        d1 += query[i + 2] * centroids[b1 >>> 4] + query[i + 3] * centroids[b1 & 0x0F];
        d2 += query[i + 4] * centroids[b2 >>> 4] + query[i + 5] * centroids[b2 & 0x0F];
        d3 += query[i + 6] * centroids[b3 >>> 4] + query[i + 7] * centroids[b3 & 0x0F];
    }
    
    // Scalar tail for any remaining dimensions
    for (; i < dim; i += 2) {
        int b = packed[i >> 1] & 0xFF;
        d0 += query[i] * centroids[b >>> 4];
        if (i + 1 < dim) {
            d0 += query[i + 1] * centroids[b & 0x0F];
        }
    }
    
    return d0 + d1 + d2 + d3;
  }

  @Override
  public float dotProductIsoQuant8Bit(byte[] packed, float[] query, float[] centroids, int dim) {
    float dot = 0f;
    for (int i = 0; i < dim; i++) {
      dot += query[i] * centroids[packed[i] & 0xFF];
    }
    return dot;
  }

  @Override
  public void byteShuffle(float[] source, byte[] dest) {
    int dim = source.length;
    for (int i = 0; i < dim; i++) {
      int bits = Float.floatToRawIntBits(source[i]);
      dest[i] = (byte) (bits >> 24);
      dest[dim + i] = (byte) (bits >> 16);
      dest[2 * dim + i] = (byte) (bits >> 8);
      dest[3 * dim + i] = (byte) bits;
    }
  }

  @Override
  public void byteUnshuffle(byte[] source, float[] dest) {
    int dim = dest.length;
    for (int i = 0; i < dim; i++) {
      int bits = ((source[i] & 0xFF) << 24)
          | ((source[dim + i] & 0xFF) << 16)
          | ((source[2 * dim + i] & 0xFF) << 8)
          | (source[3 * dim + i] & 0xFF);
      dest[i] = Float.intBitsToFloat(bits);
    }
  }

}
