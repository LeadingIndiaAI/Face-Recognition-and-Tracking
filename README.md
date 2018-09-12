# face-recognition-and-identification

place aligned/unaligned face images in dataset folder under a folder name which will same as ur label.

then run encode_faces. (This will take time depending on GPU/CPU!)

This will train the model with ur face data and generate a new encodings.pickle.

A pickle file is already uploaded trained with dataset also upoaded.

If encodings.pickle exists then face_capture can be run.

It will capture video from primary camera and detect faces logging entering and leaving time. 
