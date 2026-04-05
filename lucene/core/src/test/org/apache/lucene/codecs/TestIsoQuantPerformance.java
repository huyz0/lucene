package org.apache.lucene.codecs;

import java.io.InputStream;

public class TestIsoQuantPerformance extends BaseIsoQuantPerformanceTest {

  @Override
  protected InputStream getBaseInputStream() throws Exception {
    return TestIsoQuantPerformance.class.getResourceAsStream("siftsmall/siftsmall_base.fvecs");
  }

  @Override
  protected InputStream getQueryInputStream() throws Exception {
    return TestIsoQuantPerformance.class.getResourceAsStream("siftsmall/siftsmall_query.fvecs");
  }

  @Override
  protected InputStream getGroundtruthInputStream() throws Exception {
    return TestIsoQuantPerformance.class.getResourceAsStream("siftsmall/siftsmall_groundtruth.ivecs");
  }

  @Override
  protected String getDatasetName() {
    return "SIFT10k (siftsmall)";
  }
}
