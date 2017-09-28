# HEVC-domain-Person-Vehicle-Classification
Segmentation and classification of person and vehicles in compressed videos (HEVC/ H.265)

In video analysis at large scales, such as content analysis and search for a large surveillance network, the complexity of video decoding becomes a major bottleneck of the real-time system.<br>
To address this issue, I explored compression-domain approaches for video content analysis which extract features directly from the bit stream syntax, such as motion vectors and block coding modes.<br>
This implies low computational complexity since the full-scale decoding and reconstruction of pixels are avoided.

Features extracted from compressed bitstream: motion vectors, prediction modes, coding unit depth, DCT coefficients. Feature extraction involved extensive debugging of open source [HM software codec](https://hevc.hhi.fraunhofer.de/svn/svn_HEVCSoftware/tags/HM-16.0/) (reference implementation of the HEVC coding)<br><br>

Amongst the models tried, SVM yielded accuracy of 90% for person-vehicle classification.<br>
Also, deployed a deep learning framework that inputs frame size features processed by a four-stream spatial network. Then these four-stream features are fused in a recurrent way for learning discriminative motion contexts.
Accuracy achieved is 70% - the architecture is still being tuned to achieve higher accuracy.
