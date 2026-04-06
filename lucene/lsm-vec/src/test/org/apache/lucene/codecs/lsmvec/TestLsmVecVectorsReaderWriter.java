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

import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues.DocIndexIterator;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.Test;

public class TestLsmVecVectorsReaderWriter extends LuceneTestCase {

  @Test
  public void testFloatVectorsReadWrite() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = newIndexWriterConfig();
      // Ensure we lock the codec to LSM-VEC exclusively
      iwc.setCodec(TestUtil.alwaysKnnVectorsFormat(new LsmVecVectorsFormat()));

      try (IndexWriter iw = new IndexWriter(dir, iwc)) {
        Document doc1 = new Document();
        doc1.add(
            new KnnFloatVectorField(
                "fv", new float[] {1f, 2f, 3f}, VectorSimilarityFunction.EUCLIDEAN));
        iw.addDocument(doc1);

        Document doc2 = new Document();
        doc2.add(
            new KnnFloatVectorField(
                "fv", new float[] {4f, 5f, 6f}, VectorSimilarityFunction.EUCLIDEAN));
        iw.addDocument(doc2);
      }

      try (IndexReader reader = DirectoryReader.open(dir)) {
        assertEquals(1, reader.leaves().size());
        LeafReader leaf = reader.leaves().getFirst().reader();

        FloatVectorValues values = leaf.getFloatVectorValues("fv");
        assertNotNull(values);
        assertEquals(2, values.size());

        DocIndexIterator it = values.iterator();
        assertEquals(0, it.nextDoc());
        assertArrayEquals(new float[] {1f, 2f, 3f}, values.vectorValue(it.index()), 0.0001f);

        assertEquals(1, it.nextDoc());
        assertArrayEquals(new float[] {4f, 5f, 6f}, values.vectorValue(it.index()), 0.0001f);
      }
    }
  }

  @Test
  public void testByteVectorsReadWrite() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = newIndexWriterConfig();
      iwc.setCodec(TestUtil.alwaysKnnVectorsFormat(new LsmVecVectorsFormat()));

      try (IndexWriter iw = new IndexWriter(dir, iwc)) {
        Document doc1 = new Document();
        doc1.add(
            new KnnByteVectorField("bv", new byte[] {1, 2, 3}, VectorSimilarityFunction.COSINE));
        iw.addDocument(doc1);

        Document doc2 = new Document();
        doc2.add(
            new KnnByteVectorField("bv", new byte[] {4, 5, 6}, VectorSimilarityFunction.COSINE));
        iw.addDocument(doc2);
      }

      try (IndexReader reader = DirectoryReader.open(dir)) {
        assertEquals(1, reader.leaves().size());
        LeafReader leaf = reader.leaves().get(0).reader();

        ByteVectorValues values = leaf.getByteVectorValues("bv");
        assertNotNull(values);
        assertEquals(2, values.size());

        DocIndexIterator it = values.iterator();
        assertEquals(0, it.nextDoc());
        assertArrayEquals(new byte[] {1, 2, 3}, values.vectorValue(it.index()));

        assertEquals(1, it.nextDoc());
        assertArrayEquals(new byte[] {4, 5, 6}, values.vectorValue(it.index()));
      }
    }
  }

  @Test
  public void testSearch() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = newIndexWriterConfig();
      iwc.setCodec(TestUtil.alwaysKnnVectorsFormat(new LsmVecVectorsFormat()));

      try (IndexWriter iw = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < 100; i++) {
          Document doc = new Document();
          doc.add(
              new KnnFloatVectorField(
                  "fv",
                  new float[] {i * 0.1f, i * 0.2f, i * 0.3f},
                  VectorSimilarityFunction.EUCLIDEAN));
          iw.addDocument(doc);
        }
      }

      try (IndexReader reader = DirectoryReader.open(dir)) {
        assertEquals(1, reader.leaves().size());
        LeafReader leaf = reader.leaves().get(0).reader();

        org.apache.lucene.search.TopDocs topDocs =
            leaf.searchNearestVectors(
                "fv", new float[] {5.0f, 10.0f, 15.0f}, 10, null, Integer.MAX_VALUE);
        assertNotNull(topDocs);
        assertEquals(10, topDocs.scoreDocs.length);

        // Document 50 will perfectly match {5.0f, 10.0f, 15.0f} since 50 * 0.1 = 5.0
        assertEquals(50, topDocs.scoreDocs[0].doc);
      }
    }
  }
}
