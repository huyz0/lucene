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

import java.io.IOException;
import java.nio.FloatBuffer;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.VectorUtil;

public final class HeapLsmVecGraph extends LsmVecGraph {
  private float[][] floatVectors;
  private byte[][] byteVectors;

  public HeapLsmVecGraph(
      int maxEdges,
      int efConstruction,
      FieldInfo fieldInfo,
      int vectorDimension,
      int vectorsPerBlock) {
    super(maxEdges, efConstruction, fieldInfo, vectorDimension, vectorsPerBlock);
    this.floatVectors = new float[maxEdges][];
    this.byteVectors = new byte[maxEdges][];
  }

  @Override
  public void setVectorValue(int targetOrd, Object vectorValue) {
    if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
      if (targetOrd >= floatVectors.length) {
        float[][] newVectors = new float[Math.max(floatVectors.length * 2, targetOrd + 1)][];
        System.arraycopy(floatVectors, 0, newVectors, 0, floatVectors.length);
        floatVectors = newVectors;
      }
      floatVectors[targetOrd] = ((float[]) vectorValue).clone();
    } else {
      if (targetOrd >= byteVectors.length) {
        byte[][] newVectors = new byte[Math.max(byteVectors.length * 2, targetOrd + 1)][];
        System.arraycopy(byteVectors, 0, newVectors, 0, byteVectors.length);
        byteVectors = newVectors;
      }
      byteVectors[targetOrd] = ((byte[]) vectorValue).clone();
    }
  }

  @Override
  public void writeVectorData(
      int ord, IndexOutput vectorDataOutput, FloatBuffer floatBuffer, byte[] floatBufferBytes)
      throws IOException {
    if (encoding == VectorEncoding.FLOAT32) {
      float[] vec = floatVectors[ord];
      floatBuffer.clear();
      floatBuffer.put(vec);
      vectorDataOutput.writeBytes(floatBufferBytes, 0, floatBufferBytes.length);
    } else {
      byte[] vec = byteVectors[ord];
      vectorDataOutput.writeBytes(vec, 0, vec.length);
    }
  }

  @Override
  public float computeDistance(int aOrd, int bOrd) {
    if (encoding == VectorEncoding.FLOAT32) {
      float[] v1 = floatVectors[aOrd];
      float[] v2 = floatVectors[bOrd];
      return switch (similarityFunction) {
        case EUCLIDEAN -> VectorUtil.normalizeDistanceToUnitInterval(VectorUtil.squareDistance(v1, v2));
        case DOT_PRODUCT -> VectorUtil.normalizeToUnitInterval(VectorUtil.dotProduct(v1, v2));
        case COSINE -> VectorUtil.normalizeToUnitInterval(VectorUtil.cosine(v1, v2));
        case MAXIMUM_INNER_PRODUCT -> VectorUtil.scaleMaxInnerProductScore(VectorUtil.dotProduct(v1, v2));
        default -> throw new IllegalArgumentException("Unsupported similarity function: " + similarityFunction);
      };
    } else {
      byte[] v1 = byteVectors[aOrd];
      byte[] v2 = byteVectors[bOrd];
      return switch (similarityFunction) {
        case EUCLIDEAN -> 1.0f / (1.0f + VectorUtil.squareDistance(v1, v2));
        case DOT_PRODUCT, COSINE, MAXIMUM_INNER_PRODUCT -> VectorUtil.dotProductScore(v1, v2);
        default -> throw new IllegalArgumentException("Unsupported similarity function: " + similarityFunction);
      };
    }
  }

  @Override
  public void bulkComputeDistance(int targetOrd, int[] neighborOrds, float[] outScores, int count) {
    if (encoding == VectorEncoding.FLOAT32) {
      float[] targetVec = floatVectors[targetOrd];
      switch (similarityFunction) {
        case EUCLIDEAN:
          for (int i = 0; i < count; i++) {
            outScores[i] = VectorUtil.normalizeDistanceToUnitInterval(
                VectorUtil.squareDistance(targetVec, floatVectors[neighborOrds[i]]));
          }
          break;
        case DOT_PRODUCT:
          for (int i = 0; i < count; i++) {
            outScores[i] = VectorUtil.normalizeToUnitInterval(
                VectorUtil.dotProduct(targetVec, floatVectors[neighborOrds[i]]));
          }
          break;
        case COSINE:
          for (int i = 0; i < count; i++) {
            outScores[i] = VectorUtil.normalizeToUnitInterval(
                VectorUtil.cosine(targetVec, floatVectors[neighborOrds[i]]));
          }
          break;
        case MAXIMUM_INNER_PRODUCT:
          for (int i = 0; i < count; i++) {
            outScores[i] = VectorUtil.scaleMaxInnerProductScore(
                VectorUtil.dotProduct(targetVec, floatVectors[neighborOrds[i]]));
          }
          break;
      }
    } else {
      byte[] targetVec = byteVectors[targetOrd];
      switch (similarityFunction) {
        case EUCLIDEAN:
          for (int i = 0; i < count; i++) {
            outScores[i] = 1.0f / (1.0f + VectorUtil.squareDistance(targetVec, byteVectors[neighborOrds[i]]));
          }
          break;
        case DOT_PRODUCT:
        case COSINE:
        case MAXIMUM_INNER_PRODUCT:
          for (int i = 0; i < count; i++) {
            outScores[i] = VectorUtil.dotProductScore(targetVec, byteVectors[neighborOrds[i]]);
          }
          break;
      }
    }
  }

  @Override
  public long ramBytesUsed() {
    long bytes = super.ramBytesUsed();
    if (encoding == VectorEncoding.FLOAT32) {
      bytes += floatVectors.length * (long) org.apache.lucene.util.RamUsageEstimator.NUM_BYTES_OBJECT_REF;
      bytes += size() * (long) vectorDimension * Float.BYTES;
    } else {
      bytes += byteVectors.length * (long) org.apache.lucene.util.RamUsageEstimator.NUM_BYTES_OBJECT_REF;
      bytes += size() * (long) vectorDimension * Byte.BYTES;
    }
    return bytes;
  }
}
