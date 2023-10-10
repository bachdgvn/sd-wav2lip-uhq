import json
from scripts.wav2lip.w2l import W2l
from scripts.wav2lip.wav2lip_uhq import Wav2LipUHQ
from scripts.faceswap.swap import FaceSwap


def generate(video, face_swap_img, face_index, audio, checkpoint, face_restore_model, no_smooth, only_mouth, resize_factor,
             mouth_mask_dilatation, erode_face_mask, mask_blur, pad_top, pad_bottom, pad_left, pad_right,
             active_debug, code_former_weight):
    if video is None or audio is None:
        print("[ERROR] Please select a video and an audio file")
        return

    if face_swap_img is not None:
        face_swap = FaceSwap(video, audio, face_index, face_swap_img, resize_factor, face_restore_model, code_former_weight)
        video = face_swap.generate()

    w2l = W2l(video, audio, checkpoint, no_smooth, resize_factor, pad_top, pad_bottom, pad_left,
              pad_right, face_swap_img)
    w2l.execute()

    w2luhq = Wav2LipUHQ(video, face_restore_model, mouth_mask_dilatation, erode_face_mask, mask_blur,
                        only_mouth, face_swap_img, resize_factor, code_former_weight, active_debug)

    return w2luhq.execute()