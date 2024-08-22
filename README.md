# Brain-Tumor-Classification
This repository contains the data and code needed to train convolutional neural networks on brain tumor detection and classification, with or without camouflage animal transfer learning. Also included is the code for various XAI methods used for network analysis, such as feature spaces, DeepDreamImage, and imageSensitivity maps.
**Note**: the pretrained camouflage animal network must be accessed from the following repository (https://www.kaggle.com/datasets/farisrustom/camoanimals), since the file size exceeded GitHub's allowance. Using this network as the basis for tumor detection and classification can increase the network's performance on the task.

### How to use these files
1. Download and unzip the 4 dataset files (T1 training/testing, T2 training/testing)
2. Run the T1 and T2 training file code on the data to train the networks
3. Run the code from the bottom section of this same file to generate feature spaces of the resulting networks, showing their internal representation of the data
4. Download the pretrained camouflage animal detection network for transfer learning, from the following link: https://www.kaggle.com/datasets/farisrustom/camoanimals
5. Run the T1 and T2 with Camo Transfer Learning file code to use the weights from camouflage animal training for tumor detection
6. Run the code to generate feature spaces of these network, from the same file above
7. Use the code in the GliomaDDI file to generate DeepDreamImages of the trained networks. You can modify which layers of the network's architecture the images are generated from
8. Use the code in imageSensitivity functions Part 1&2 to generate gradCAM and occlusion sensitivity maps of the trained networks. Again, you can modify which layers of the network these sensitivity maps are based upon
