import cv2
import face_recognition
import os


def load_encode_names(folder: str = 'data/datasets/known-faces'):
    """Get names of known faces

    Returns:
        list: Known faces list
    """
    list_names = []
    for filename in os.listdir(folder):
        list_names.append(filename)
    return list_names

def enconding_face(path: str, locations: bool = False):
    face = face_recognition.load_image_file(path)
    face_encodings = face_recognition.face_encodings(face)
    face_locations = face_recognition.face_locations(face) if locations == True else None
    return face_encodings, face, face_locations

def load_encode_faces(folder: str = 'data/datasets/known-faces'):
    known_encode_faces = []
    for filename in os.listdir(folder):
        known_encode_face = enconding_face(f'{folder}/{filename}')[0][0]
        known_encode_faces.append(known_encode_face)
    return known_encode_faces

def display_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

known_face_names = load_encode_names()
known_face_encodings = load_encode_faces()

# check_image = 'data/test/celedion/celedion-5.jpg'
# check_image = 'data/test/taylor/taylor-5.jpg'
check_image = 'data/test/taylor-in-group-1.jpg'
check_face_encodings, check_face, check_face_locations = enconding_face(check_image, locations=True)

for (top, right, bottom, left), check_face_encoding in zip(check_face_locations, check_face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, check_face_encoding, tolerance=0.55)
    print(matches)
    name = 'unknown'
    if True in matches:
        match_index = matches.index(True)
        name = known_face_names[match_index]

        # Only shown when detected
        cv2.rectangle(check_face, (left, top), (right, bottom), (255, 0, 0), 3)
        cv2.rectangle(check_face, (left, bottom+30), (right, bottom), (255, 0, 0), cv2.FILLED)
        cv2.putText(check_face, name, (left+3, bottom+25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

display_face(check_face)