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
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestLsmVecIntegration extends LuceneTestCase {

  public void testBasicIndexingAndSearch() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = newIndexWriterConfig();
      // Set our new codec
      iwc.setCodec(new LsmVecCodec());

      try (IndexWriter writer = new IndexWriter(dir, iwc)) {
        // Doc 1
        Document doc1 = new Document();
        doc1.add(new StringField("id", "1", Field.Store.YES));
        doc1.add(
            new KnnFloatVectorField(
                "vector", new float[] {1f, 0f, 0f}, VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc1);

        // Doc 2
        Document doc2 = new Document();
        doc2.add(new StringField("id", "2", Field.Store.YES));
        doc2.add(
            new KnnFloatVectorField(
                "vector", new float[] {0f, 1f, 0f}, VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc2);

        // Doc 3
        Document doc3 = new Document();
        doc3.add(new StringField("id", "3", Field.Store.YES));
        doc3.add(
            new KnnFloatVectorField(
                "vector", new float[] {0f, 0f, 1f}, VectorSimilarityFunction.EUCLIDEAN));
        writer.addDocument(doc3);

        writer.commit();
      }

      // Verify search
      try (IndexReader reader = DirectoryReader.open(dir)) {
        IndexSearcher searcher = new IndexSearcher(reader);

        // Search exactly for doc 2 vector
        float[] queryVector = new float[] {0f, 0.9f, 0.1f};
        KnnFloatVectorQuery query = new KnnFloatVectorQuery("vector", queryVector, 1);

        TopDocs topDocs = searcher.search(query, 5);
        assertEquals(1, topDocs.scoreDocs.length);

        Document hitDoc = searcher.storedFields().document(topDocs.scoreDocs[0].doc);
        assertEquals("2", hitDoc.get("id"));
      }
    }
  }
}
