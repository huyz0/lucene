package org.apache.lucene.codecs;

import java.io.InputStream;
import org.junit.Assume;

public class TestIsoQuantPerformanceSift200k extends BaseIsoQuantPerformanceTest {

  @Override
  public void setUp() throws Exception {
    Assume.assumeTrue(
        "SIFT200k dataset is not available",
        TestIsoQuantPerformanceSift200k.class.getResource("sift200k_sampled/sift_base.fvecs") != null);
    super.setUp();
  }

  @Override
  protected InputStream getBaseInputStream() throws Exception {
    return TestIsoQuantPerformanceSift200k.class.getResourceAsStream("sift200k_sampled/sift_base.fvecs");
  }

  @Override
  protected InputStream getQueryInputStream() throws Exception {
    return TestIsoQuantPerformanceSift200k.class.getResourceAsStream("sift200k_sampled/sift_query.fvecs");
  }

  @Override
  protected InputStream getGroundtruthInputStream() throws Exception {
    return TestIsoQuantPerformanceSift200k.class.getResourceAsStream("sift200k_sampled/sift_groundtruth.ivecs");
  }

  @Override
  protected String getDatasetName() {
    return "SIFT200k";
  }
}
