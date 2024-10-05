# SHAZAM-CAPSTONE
UAA fall 2024 CAPSTONE project

Members: David Kim, Joe Groth, Uriah A

**Project Description**

Music discovery has never been easier with apps like Shazam that listens to audio to provide back with information about that song. In this project, over 188 million Shazam query timings, aggregated across 20 songs will be analyzed to create histograms of user queries for each song. We hypothesize that users are more likely to initiate a Shazam query during significant musical events, such as the chorus or a unique instrumental section. By leveraging machine learning, we aim to predict the probability of a Shazam query based on the audio features of a song. These predictions will help identify patterns in user behavior based on the music's structure.

**Requirements**

Open Source Tool-Kits:

MSAF (Music Structure Analysis Framework) - tool specifically designed for segmenting the structure of music, breaking a musical piece into meaningful sections such as verses, choruses, and bridges.

https://msaf.readthedocs.io/en/latest/

https://github.com/urinieto/msaf

librosa - Python library designed for music and audio analysis. It provides tools for various tasks in music information retrieval (MIR) and digital signal processing.

https://librosa.org/doc/latest/index.html#

https://github.com/librosa/librosa

**Installation Instructions**

MSAF:

1. create conda env -> conda create -n msaf_env python=3.7
2. download and git clone https://github.com/urinieto/msaf.git
3. open requirements.txt and change pandas version to 1.1.5
4. conda install numpy scipy
5. conda install scikit-learn
6. conda install -c conda-forge cvxopt
7. pip install .


**References**

Kaneshiro, Blair, et al. “Characterizing listener engagement with popular songs using large-scale music discovery data.” Frontiers in Psychology, vol. 8, 23 Mar. 2017, https://doi.org/10.3389/fpsyg.2017.00416.

Kaneshiro, Blair, et al. “Characterizing Listener Engagement with Popular Songs Using Large-Scale Music Discovery Data.” Frontiers in Psychology, vol. 8, 2017, https://doi.org/10.3389/fpsyg.2017.00416.

McFee, Brian, et al. “Librosa: Audio and Music Signal Analysis in python.” Proceedings of the Python in Science Conference, 2015, pp. 18–24, https://doi.org/10.25080/majora-7b98e3ed-003.

Nieto, Oriol. “SYSTEMATIC EXPLORATION OF COMPUTATIONAL MUSIC  STRUCTURE RESEARCH.” Center for Computer Research in Music and Acoustics, Stanford University, Aug. 2016, ccrma.stanford.edu/~urinieto/MARL/publications/ISMIR2016-NietoBello.pdf.

Tzanetakis, G., and P. Cook. “Musical genre classification of Audio Signals.” IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, July 2002, pp. 293–302, https://doi.org/10.1109/tsa.2002.800560. 

**Installation**


**Usage Instructions**


