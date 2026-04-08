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
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene95.OrdToDocDISIReaderConfiguration;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.MergeState.DocMap;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;

public final class LsmVecVectorsWriter extends KnnVectorsWriter {

  private final IndexOutput metaOutput;
  private final IndexOutput vectorEdgeOutput;
  private final IndexOutput vectorDataOutput;

  private final List<LsmVecFieldWriter<?>> fields = new ArrayList<>();

  private final int efConstruction;
  private final int maxEdges;

  public LsmVecVectorsWriter(SegmentWriteState state, int maxEdges, int efConstruction)
      throws IOException {
    this.efConstruction = efConstruction;
    this.maxEdges = maxEdges;

    boolean success = false;
    IndexOutput metaOut = null;
    IndexOutput edgeOut = null;
    IndexOutput dataOut = null;
    try {
      metaOut =
          state.directory.createOutput(
              state.segmentInfo.name + "." + LsmVecVectorsFormat.VEM_META_EXTENSION, state.context);
      this.metaOutput = metaOut;
      CodecUtil.writeIndexHeader(
          metaOut, "LsmVecMeta", 1, state.segmentInfo.getId(), state.segmentSuffix);

      edgeOut =
          state.directory.createOutput(
              state.segmentInfo.name + "." + LsmVecVectorsFormat.VEM_EXTENSION, state.context);
      this.vectorEdgeOutput = edgeOut;
      CodecUtil.writeIndexHeader(
          edgeOut, "LsmVecEdge", 1, state.segmentInfo.getId(), state.segmentSuffix);

      dataOut =
          state.directory.createOutput(
              state.segmentInfo.name + "." + LsmVecVectorsFormat.VECD_EXTENSION, state.context);
      this.vectorDataOutput = dataOut;
      CodecUtil.writeIndexHeader(
          dataOut, "LsmVecData", 1, state.segmentInfo.getId(), state.segmentSuffix);

      success = true;
    } finally {
      if (!success) {
        IOUtils.closeWhileHandlingException(metaOut, edgeOut, dataOut);
      }
    }
  }

  @Override
  public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) {
    LsmVecFieldWriter<?> writer = new LsmVecFieldWriter<>(fieldInfo, maxEdges, efConstruction);
    fields.add(writer);
    return writer;
  }

  @Override
  public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
    for (LsmVecFieldWriter<?> fieldWriter : fields) {
      if (fieldWriter.isFinished()) {
        continue;
      }
      fieldWriter.flushGraph(vectorEdgeOutput, vectorDataOutput, metaOutput, maxDoc, sortMap);
    }
  }

  @SuppressWarnings("unchecked")
  @Override
  public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
    LsmVecFieldWriter<?> writer = new LsmVecFieldWriter<>(fieldInfo, maxEdges, efConstruction);

    int largestReaderIndex = -1;
    long largestReaderDocs = -1;
    LsmVecVectorsReader largestReader = null;

    int maxDoc = mergeState.segmentInfo.maxDoc();
    int[] newDocToSegment = new int[maxDoc];
    long[] newDocToOldOrd = new long[maxDoc];
    java.util.Arrays.fill(newDocToSegment, -1);

    int[][] segmentOldToNewOrds = new int[mergeState.knnVectorsReaders.length][];
    int[] segmentSizes = new int[mergeState.knnVectorsReaders.length];
    long[] segmentEdgePtrs = new long[mergeState.knnVectorsReaders.length];
    long nodeBytes = Integer.highestOneBit((Integer.BYTES + maxEdges * Integer.BYTES) - 1) << 1;
    IndexInput[] edgeInputs = new IndexInput[mergeState.knnVectorsReaders.length];

    for (int i = 0; i < mergeState.knnVectorsReaders.length; i++) {
      KnnVectorsReader reader = mergeState.knnVectorsReaders[i];
      if (reader == null) continue;

      reader = reader.unwrapReaderForField(fieldInfo.name);
      if (reader instanceof LsmVecVectorsReader lsmReader) {
        LsmVecVectorsReader.FieldEntry entry = lsmReader.getFieldEntry(fieldInfo.name);
        if (entry != null) {
          segmentOldToNewOrds[i] = new int[entry.size];
          java.util.Arrays.fill(segmentOldToNewOrds[i], -1);
          segmentSizes[i] = entry.size;
          segmentEdgePtrs[i] = entry.edgePtr;
        }
        edgeInputs[i] = lsmReader.getVectorEdgeInput().clone();

        DocMap docMap = mergeState.docMaps[i];

        int count = 0;
        KnnVectorValues oldValues =
            fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32
                ? lsmReader.getFloatVectorValues(fieldInfo.name)
                : lsmReader.getByteVectorValues(fieldInfo.name);

        if (oldValues == null) continue;
        KnnVectorValues.DocIndexIterator oldIter = oldValues.iterator();
        for (int oldDoc = oldIter.nextDoc();
            oldDoc != DocIdSetIterator.NO_MORE_DOCS;
            oldDoc = oldIter.nextDoc()) {
          int newDoc = docMap.get(oldDoc);
          if (newDoc != -1) {
            newDocToSegment[newDoc] = i;
            newDocToOldOrd[newDoc] = oldIter.index();
            count++;
          }
        }
        if (count > largestReaderDocs) {
          largestReaderDocs = count;
          largestReader = lsmReader;
          largestReaderIndex = i;
        }
      }
    }

    if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
      FloatVectorValues mergedFloats =
          MergedVectorValues.mergeFloatVectorValues(fieldInfo, mergeState);
      KnnVectorValues.DocIndexIterator iter = mergedFloats.iterator();
      for (int doc = iter.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iter.nextDoc()) {
        float[] vectorValue = mergedFloats.vectorValue(iter.index());
        int segIndex = newDocToSegment[doc];
        int oldOrd = -1;
        if (segIndex != -1) {
          oldOrd = (int) newDocToOldOrd[doc];
          if (segmentOldToNewOrds[segIndex] != null) {
            segmentOldToNewOrds[segIndex][oldOrd] = writer.currentOrd;
          }
        }

        int[] hints = null;
        if (segIndex != -1 && oldOrd < segmentSizes[segIndex]) {
            IndexInput edgeInput = edgeInputs[segIndex];
            edgeInput.seek(segmentEdgePtrs[segIndex] + (long) oldOrd * nodeBytes);
            int numNeighbors = edgeInput.readInt();
            hints = new int[numNeighbors];
            int validHintsCount = 0;
            for (int n = 0; n < numNeighbors; n++) {
                int neighborOldOrd = edgeInput.readInt();
                int neighborNewOrd = segmentOldToNewOrds[segIndex][neighborOldOrd];
                if (neighborNewOrd != -1 && neighborNewOrd < writer.currentOrd) {
                    hints[validHintsCount++] = neighborNewOrd;
                }
            }
            if (validHintsCount > 0) {
                hints = java.util.Arrays.copyOf(hints, validHintsCount);
            } else if (writer.currentOrd > 0) {
                hints = new int[] { writer.currentOrd - 1 };
            }
        }
        if (hints != null && segIndex == largestReaderIndex) {
            ((LsmVecFieldWriter<float[]>) writer).enqueueAndAddNodeWithoutSearch(doc, vectorValue);
            for (int neighborOrd : hints) {
                float dist = writer.graph.computeDistance(writer.currentOrd - 1, neighborOrd);
                writer.graph.injectEdge(writer.currentOrd - 1, neighborOrd, dist);
            }
        } else if (hints != null) {
            ((LsmVecFieldWriter<float[]>) writer).enqueueAndAddNodeWithHints(doc, vectorValue, hints);
        } else if (writer.currentOrd > 0) {
            hints = new int[] { writer.currentOrd - 1 };
            ((LsmVecFieldWriter<float[]>) writer).enqueueAndAddNodeWithHints(doc, vectorValue, hints);
        } else {
            ((LsmVecFieldWriter<float[]>) writer).enqueueAndAddNode(doc, vectorValue);
        }
      }
    } else {
      ByteVectorValues mergedBytes =
          MergedVectorValues.mergeByteVectorValues(fieldInfo, mergeState);
      KnnVectorValues.DocIndexIterator iter = mergedBytes.iterator();
      for (int doc = iter.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iter.nextDoc()) {
        byte[] vectorValue = mergedBytes.vectorValue(iter.index());
        int segIndex = newDocToSegment[doc];
        int oldOrd = -1;
        if (segIndex != -1) {
          oldOrd = (int) newDocToOldOrd[doc];
          if (segmentOldToNewOrds[segIndex] != null) {
            segmentOldToNewOrds[segIndex][oldOrd] = writer.currentOrd;
          }
        }

        int[] hints = null;
        if (segIndex != -1 && oldOrd < segmentSizes[segIndex]) {
            IndexInput edgeInput = edgeInputs[segIndex];
            edgeInput.seek(segmentEdgePtrs[segIndex] + (long) oldOrd * nodeBytes);
            int numNeighbors = edgeInput.readInt();
            hints = new int[numNeighbors];
            int validHintsCount = 0;
            for (int n = 0; n < numNeighbors; n++) {
                int neighborOldOrd = edgeInput.readInt();
                int neighborNewOrd = segmentOldToNewOrds[segIndex][neighborOldOrd];
                if (neighborNewOrd != -1 && neighborNewOrd < writer.currentOrd) {
                    hints[validHintsCount++] = neighborNewOrd;
                }
            }
            if (validHintsCount > 0) {
                hints = java.util.Arrays.copyOf(hints, validHintsCount);
            } else if (writer.currentOrd > 0) {
                hints = new int[] { writer.currentOrd - 1 };
            }
        }
        if (hints != null && segIndex == largestReaderIndex) {
            ((LsmVecFieldWriter<byte[]>) writer).enqueueAndAddNodeWithoutSearch(doc, vectorValue);
            for (int neighborOrd : hints) {
                float dist = writer.graph.computeDistance(writer.currentOrd - 1, neighborOrd);
                writer.graph.injectEdge(writer.currentOrd - 1, neighborOrd, dist);
            }
        } else if (hints != null) {
            ((LsmVecFieldWriter<byte[]>) writer).enqueueAndAddNodeWithHints(doc, vectorValue, hints);
        } else if (writer.currentOrd > 0) {
            hints = new int[] { writer.currentOrd - 1 };
            ((LsmVecFieldWriter<byte[]>) writer).enqueueAndAddNodeWithHints(doc, vectorValue, hints);
        } else {
            ((LsmVecFieldWriter<byte[]>) writer).enqueueAndAddNode(doc, vectorValue);
        }
      }
    }



    writer.flushGraph(
        vectorEdgeOutput, vectorDataOutput, metaOutput, mergeState.segmentInfo.maxDoc(), null);
  }

  @Override
  public void finish() throws IOException {
    CodecUtil.writeFooter(metaOutput);
    CodecUtil.writeFooter(vectorEdgeOutput);
    CodecUtil.writeFooter(vectorDataOutput);
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(metaOutput, vectorEdgeOutput, vectorDataOutput);
  }

  @Override
  public long ramBytesUsed() {
    long bytes = 0;
    for (LsmVecFieldWriter<?> field : fields) {
      bytes += field.ramBytesUsed();
    }
    return bytes;
  }

  private static class LsmVecFieldWriter<T> extends KnnFieldVectorsWriter<T> {

    private final FieldInfo fieldInfo;
    private final LsmVecGraph graph;
    private boolean finished;
    private int currentOrd = 0;
    private int lastDocID = -1;

    private final int maxEdges;
    private final DocsWithFieldSet docsWithField;

    LsmVecFieldWriter(FieldInfo fieldInfo, int maxEdges, int efConstruction) {
      this.fieldInfo = fieldInfo;
      this.maxEdges = maxEdges;
      this.docsWithField = new DocsWithFieldSet();

      int dim = fieldInfo.getVectorDimension();
      int byteSizePerVector = dim
          * (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32
          ? Float.BYTES
          : Byte.BYTES);

      // Dynamically size chunks to target ~1MB per block allocation natively
      int vectorsPerBlock = Math.max(1, (1024 * 1024) / byteSizePerVector);

      this.graph = LsmVecGraphProvider.getInstance()
          .getGraph(maxEdges, efConstruction, fieldInfo, dim, vectorsPerBlock);
    }

    public void enqueueAndAddNodeWithoutSearch(int docID, T vectorValue) throws IOException {
      int targetOrd = currentOrd++;
      docsWithField.add(docID);

      graph.setVectorValue(targetOrd, vectorValue);
      graph.addNodeWithoutSearch(targetOrd);
    }

    public void enqueueAndAddNodeWithHints(int docID, T vectorValue, int[] hints) throws IOException {
      int targetOrd = currentOrd++;
      docsWithField.add(docID);

      graph.setVectorValue(targetOrd, vectorValue);
      graph.addNodeWithHints(targetOrd, targetOrd, hints);
    }

    public void enqueueAndAddNode(int docID, T vectorValue) throws IOException {
      int targetOrd = currentOrd++;
      docsWithField.add(docID);

      graph.setVectorValue(targetOrd, vectorValue);
      graph.addNode(targetOrd, targetOrd);
    }

    @Override
    public void addValue(int docID, T vectorValue) throws IOException {
      if (docID <= lastDocID) {
        throw new IllegalArgumentException(
            "VectorValuesField \""
                + fieldInfo.name
                + "\" appears more than once in this document (only one value is allowed per field)");
      }
      lastDocID = docID;
      enqueueAndAddNode(docID, vectorValue);
    }

    @Override
    public T copyValue(T vectorValue) {
      throw new UnsupportedOperationException();
    }

    void flushGraph(
        IndexOutput vectorEdgeOutput,
        IndexOutput vectorDataOutput,
        IndexOutput metaOutput,
        int maxDoc,
        Sorter.DocMap sortMap)
        throws IOException {

      // Serialize Adjacency List spanning Ordinals tightly
      int totalOrds = currentOrd;
      long edgePtr = vectorEdgeOutput.getFilePointer();
      int[] neighborBuffer = new int[maxEdges];

      int[] oldOrdToNewOrd = null;
      int[] newOrdToOldOrd = null;
      DocsWithFieldSet finalDocsWithField = this.docsWithField;

      if (sortMap != null) {
        oldOrdToNewOrd = new int[totalOrds];
        newOrdToOldOrd = new int[totalOrds];

        long[] newDocAndOldOrd = new long[totalOrds];
        DocIdSetIterator iter = docsWithField.iterator();
        for (int oldOrd = 0; oldOrd < totalOrds; oldOrd++) {
          int oldDocID = iter.nextDoc();
          int newDocID = sortMap.oldToNew(oldDocID);
          newDocAndOldOrd[oldOrd] = (((long) newDocID) << 32) | (oldOrd & 0xFFFFFFFFL);
        }
        java.util.Arrays.sort(newDocAndOldOrd);

        finalDocsWithField = new DocsWithFieldSet();
        for (int newOrd = 0; newOrd < totalOrds; newOrd++) {
          int newDocID = (int) (newDocAndOldOrd[newOrd] >>> 32);
          int oldOrd = (int) newDocAndOldOrd[newOrd];
          finalDocsWithField.add(newDocID);
          newOrdToOldOrd[newOrd] = oldOrd;
          oldOrdToNewOrd[oldOrd] = newOrd;
        }
      }

      int nodeBytes = Integer.highestOneBit((Integer.BYTES + maxEdges * Integer.BYTES) - 1) << 1;
      byte[] paddingBytesArr = new byte[nodeBytes];
      java.util.Arrays.fill(paddingBytesArr, (byte) 0xFF);

      for (int ord = 0; ord < totalOrds; ord++) {
        int oldOrd = newOrdToOldOrd != null ? newOrdToOldOrd[ord] : ord;
        int validEdges = graph.getSortedNeighbors(oldOrd, neighborBuffer);
        vectorEdgeOutput.writeInt(validEdges);
        for (int i = 0; i < validEdges; i++) {
          int neighborOldOrd = neighborBuffer[i];
          int neighborNewOrd =
              oldOrdToNewOrd != null ? oldOrdToNewOrd[neighborOldOrd] : neighborOldOrd;
          vectorEdgeOutput.writeInt(neighborNewOrd);
        }
        int paddingBytes = nodeBytes - (Integer.BYTES + validEdges * Integer.BYTES);
        if (paddingBytes > 0) {
            vectorEdgeOutput.writeBytes(paddingBytesArr, 0, paddingBytes);
        }
      }

      metaOutput.writeInt(fieldInfo.number);

      // Write Data Vectors sequentially
      if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
        vectorDataOutput.alignFilePointer(64);
      } else {
        vectorDataOutput.alignFilePointer(Float.BYTES);
      }
      long vectorDataOffset = vectorDataOutput.getFilePointer();

      byte[] floatBufferBytes = null;
      java.nio.FloatBuffer floatBuffer = null;
      if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
        floatBufferBytes = new byte[fieldInfo.getVectorDimension() * Float.BYTES];
        floatBuffer =
            java.nio.ByteBuffer.wrap(floatBufferBytes)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer();
      }

      for (int i = 0; i < currentOrd; i++) {
        int oldOrd = newOrdToOldOrd != null ? newOrdToOldOrd[i] : i;
        if (fieldInfo.getVectorEncoding() == VectorEncoding.FLOAT32) {
          graph.writeVectorData(oldOrd, vectorDataOutput, floatBuffer, floatBufferBytes);
        } else {
          graph.writeVectorData(oldOrd, vectorDataOutput, null, null);
        }
      }
      long vectorDataLength = vectorDataOutput.getFilePointer() - vectorDataOffset;

      metaOutput.writeLong(edgePtr);
      metaOutput.writeVLong(vectorDataOffset);
      metaOutput.writeVLong(vectorDataLength);
      metaOutput.writeInt(totalOrds);
      metaOutput.writeInt(maxEdges);
      metaOutput.writeVInt(fieldInfo.getVectorDimension());
      metaOutput.writeInt(finalDocsWithField.cardinality());
      OrdToDocDISIReaderConfiguration.writeStoredMeta(
          6,
          metaOutput,
          vectorDataOutput,
          finalDocsWithField.cardinality(),
          maxDoc,
          finalDocsWithField);

      finished = true;
    }

    boolean isFinished() {
      return finished;
    }

    @Override
    public long ramBytesUsed() {
      return graph.ramBytesUsed() + docsWithField.ramBytesUsed();
    }
  }
}
