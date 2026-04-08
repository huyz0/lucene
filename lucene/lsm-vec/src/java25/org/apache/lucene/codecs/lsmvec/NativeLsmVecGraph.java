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
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.VectorUtilFFM;

@SuppressWarnings("restricted")
public final class NativeLsmVecGraph extends LsmVecGraph {

  private final Arena arena;
  public List<MemorySegment> blocks;

  public NativeLsmVecGraph(
      int maxEdges,
      int efConstruction,
      FieldInfo fieldInfo,
      int vectorDimension,
      int vectorsPerBlock) {
    super(maxEdges, efConstruction, fieldInfo, vectorDimension, vectorsPerBlock);
    this.arena = Arena.ofAuto();
    this.blocks = new ArrayList<>();
  }

  @Override
  public void setVectorValue(int targetOrd, Object vectorValue) {
    int blockIndex = targetOrd / vectorsPerBlock;
    int offsetInBlock = targetOrd % vectorsPerBlock;

    if (encoding == VectorEncoding.FLOAT32) {
      if (blockIndex >= blocks.size()) {
        blocks.add(
            arena.allocate(
                (long) vectorsPerBlock * vectorDimension * Float.BYTES,
                ValueLayout.JAVA_FLOAT.byteAlignment()));
      }
      float[] src = (float[]) vectorValue;
      MemorySegment destBlock = blocks.get(blockIndex);
      MemorySegment.copy(
          src,
          0,
          destBlock,
          ValueLayout.JAVA_FLOAT,
          (long) offsetInBlock * vectorDimension,
          vectorDimension);
    } else {
      if (blockIndex >= blocks.size()) {
        blocks.add(
            arena.allocate(
                (long) vectorsPerBlock * vectorDimension, ValueLayout.JAVA_BYTE.byteAlignment()));
      }
      byte[] src = (byte[]) vectorValue;
      MemorySegment destBlock = blocks.get(blockIndex);
      MemorySegment.copy(
          src,
          0,
          destBlock,
          ValueLayout.JAVA_BYTE,
          (long) offsetInBlock * vectorDimension,
          vectorDimension);
    }
  }

  @Override
  public void writeVectorData(
      int ord, IndexOutput vectorDataOutput, FloatBuffer floatBuffer, byte[] floatBufferBytes)
      throws IOException {
    int blockIndex = ord / vectorsPerBlock;
    int offsetInBlock = ord % vectorsPerBlock;
    MemorySegment block = blocks.get(blockIndex);

    if (encoding == VectorEncoding.FLOAT32) {
      MemorySegment slice =
          block.asSlice(
              (long) offsetInBlock * vectorDimension * Float.BYTES,
              (long) vectorDimension * Float.BYTES);
      float[] raw = slice.toArray(ValueLayout.JAVA_FLOAT);
      floatBuffer.clear();
      floatBuffer.put(raw);
      vectorDataOutput.writeBytes(floatBufferBytes, 0, floatBufferBytes.length);
    } else {
      MemorySegment slice = block.asSlice((long) offsetInBlock * vectorDimension, vectorDimension);
      byte[] raw = slice.toArray(ValueLayout.JAVA_BYTE);
      vectorDataOutput.writeBytes(raw, 0, raw.length);
    }
  }

  @Override
  public float computeDistance(int aOrd, int bOrd) {
    int aBlockId = aOrd / vectorsPerBlock;
    int aBlockOff = (aOrd % vectorsPerBlock) * vectorDimension;

    int bBlockId = bOrd / vectorsPerBlock;
    int bBlockOff = (bOrd % vectorsPerBlock) * vectorDimension;

    MemorySegment blockA = blocks.get(aBlockId);
    MemorySegment blockB = blocks.get(bBlockId);

    if (encoding == VectorEncoding.FLOAT32) {
      MemorySegment sliceA =
          blockA.asSlice((long) aBlockOff * Float.BYTES, (long) vectorDimension * Float.BYTES);
      MemorySegment sliceB =
          blockB.asSlice((long) bBlockOff * Float.BYTES, (long) vectorDimension * Float.BYTES);

      if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
        float score = VectorUtilFFM.squareDistanceFloat(sliceA, sliceB);
        return 1.0f / (1.0f + score);
      } else if (similarityFunction == VectorSimilarityFunction.COSINE
          || similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
        float score = VectorUtilFFM.dotProductFloat(sliceA, sliceB);
        if (similarityFunction == VectorSimilarityFunction.COSINE) {
          return Math.max((1.0f + score) / 2.0f, 0.0f);
        }
        return score;
      } else if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
        float score = VectorUtilFFM.dotProductFloat(sliceA, sliceB);
        if (score < 0) return 1.0f / (1.0f - score);
        return score + 1.0f;
      }
    } else {
      throw new UnsupportedOperationException(
          "Native byte vector similarity function natively unsupported directly from MemorySegments!");
    }
    throw new UnsupportedOperationException("Similarity function natively unsupported!");
  }

  @Override
  public void bulkComputeDistance(int targetOrd, int[] neighborOrds, float[] outScores, int count) {
    if (encoding == VectorEncoding.FLOAT32) {
      int targetBlockId = targetOrd / vectorsPerBlock;
      int targetBlockOff = (targetOrd % vectorsPerBlock) * vectorDimension;
      MemorySegment targetBlock = blocks.get(targetBlockId);
      MemorySegment targetSlice = targetBlock.asSlice((long) targetBlockOff * Float.BYTES, (long) vectorDimension * Float.BYTES);

      if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
        for (int i = 0; i < count; i++) {
          int bOrd = neighborOrds[i];
          int bBlockId = bOrd / vectorsPerBlock;
          int bBlockOff = (bOrd % vectorsPerBlock) * vectorDimension;
          MemorySegment sliceB = blocks.get(bBlockId).asSlice((long) bBlockOff * Float.BYTES, (long) vectorDimension * Float.BYTES);
          float score = VectorUtilFFM.squareDistanceFloat(targetSlice, sliceB);
          outScores[i] = 1.0f / (1.0f + score);
        }
      } else if (similarityFunction == VectorSimilarityFunction.COSINE) {
        for (int i = 0; i < count; i++) {
          int bOrd = neighborOrds[i];
          int bBlockId = bOrd / vectorsPerBlock;
          int bBlockOff = (bOrd % vectorsPerBlock) * vectorDimension;
          MemorySegment sliceB = blocks.get(bBlockId).asSlice((long) bBlockOff * Float.BYTES, (long) vectorDimension * Float.BYTES);
          float score = VectorUtilFFM.dotProductFloat(targetSlice, sliceB);
          outScores[i] = Math.max((1.0f + score) / 2.0f, 0.0f);
        }
      } else if (similarityFunction == VectorSimilarityFunction.DOT_PRODUCT) {
        for (int i = 0; i < count; i++) {
          int bOrd = neighborOrds[i];
          int bBlockId = bOrd / vectorsPerBlock;
          int bBlockOff = (bOrd % vectorsPerBlock) * vectorDimension;
          MemorySegment sliceB = blocks.get(bBlockId).asSlice((long) bBlockOff * Float.BYTES, (long) vectorDimension * Float.BYTES);
          outScores[i] = VectorUtilFFM.dotProductFloat(targetSlice, sliceB);
        }
      } else if (similarityFunction == VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT) {
        for (int i = 0; i < count; i++) {
          int bOrd = neighborOrds[i];
          int bBlockId = bOrd / vectorsPerBlock;
          int bBlockOff = (bOrd % vectorsPerBlock) * vectorDimension;
          MemorySegment sliceB = blocks.get(bBlockId).asSlice((long) bBlockOff * Float.BYTES, (long) vectorDimension * Float.BYTES);
          float score = VectorUtilFFM.dotProductFloat(targetSlice, sliceB);
          if (score < 0) outScores[i] = 1.0f / (1.0f - score);
          else outScores[i] = score + 1.0f;
        }
      } else {
        throw new UnsupportedOperationException("Similarity function natively unsupported!");
      }
    } else {
      throw new UnsupportedOperationException("Native byte vector similarity function natively unsupported directly from MemorySegments!");
    }
  }

  @Override
  public long ramBytesUsed() {
    long bytes = super.ramBytesUsed();
    long bytesPerBlock = encoding == VectorEncoding.FLOAT32 ?
        (long) vectorsPerBlock * vectorDimension * Float.BYTES :
        (long) vectorsPerBlock * vectorDimension * Byte.BYTES;
    bytes += (long) blocks.size() * bytesPerBlock;
    return bytes;
  }
}

