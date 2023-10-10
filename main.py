from scripts.main import generate

generate(
    video='content/drive/MyDrive/wav2lip/sources/book.mp4',
    face_swap_img=None,
    face_index_slider=0,
    audio='content/drive/MyDrive/wav2lip/sources/thao-trinh.mp4',
    checkpoint='wav2lip', # ["wav2lip", "wav2lip_gan"]
    face_restore_model='GFPGAN', #["CodeFormer", "GFPGAN"]
    no_smooth=False,
    only_mouth=False,
    resize_factor=1,
    mouth_mask_dilatation=15, #1-128
    erode_face_mask=15, #1-128
    mask_blur=15, #1-128
    pad_top=0,#0-50
    pad_bottom=0,#0-50
    pad_left=0,#0-50
    pad_right=0,#0-50
    active_debug=False,
    code_former_weight=0.75
)
