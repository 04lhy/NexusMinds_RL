import h5py
import numpy as np
import cv2
import os

HDF5_PATH = "/home/gu/RL_agent/dataset.hdf5"   # 🔴 改成你的路径
SAVE_DIR = "check_img"

os.makedirs(SAVE_DIR, exist_ok=True)


def save_image(img, path):
    img = np.array(img)

    # -------- 维度处理 --------
    if img.ndim == 5:
        img = img.reshape(-1, *img.shape[-3:])
        img = np.transpose(img, (0, 2, 3, 1))

    elif img.ndim == 4:
        if img.shape[1] in [1, 3] and img.shape[-1] not in [1, 3]:
            img = np.transpose(img, (0, 2, 3, 1))

    # -------- 单帧处理 --------
    if img.ndim == 4:
        img = img[0]

    # ✅ 🔴 这里加：原始数据范围
    print(f"[FRAME RAW] shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")

    # -------- 归一化 --------
    if img.dtype != np.uint8:
        img = img.astype(np.float32)

        img_min = img.min()
        img_max = img.max()

        print(f"[FRAME BEFORE NORM] min={img_min}, max={img_max}")

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        img = (img * 255).clip(0, 255).astype(np.uint8)

        # ✅ 🔴 归一化后范围
        print(f"[FRAME AFTER NORM] min={img.min()}, max={img.max()}")

    # -------- 通道处理 --------
    if img.ndim == 2:
        pass

    elif img.ndim == 3:
        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        elif img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        else:
            raise ValueError(f"Invalid channel number: {img.shape}")

    else:
        raise ValueError(f"Invalid image shape: {img.shape}")

    # -------- 保存 --------
    cv2.imwrite(path, img)

def save_images_from_hdf5(hdf5_path, key="agentview_head_image"):
    with h5py.File(hdf5_path, "r") as f:

        def recursive_search(name, obj):
            if isinstance(obj, h5py.Dataset) and key in name:
                print(f"[FOUND] {name} | shape={obj.shape}")

                data = obj[()]

                # (50, 240, 320, 3)
                print(f"[INFO] raw data shape: {data.shape}, dtype: {data.dtype}")

                # -------- 保存每一帧 --------
                for i in range(data.shape[0]):
                    frame = data[i]

                    save_path = os.path.join(
                        SAVE_DIR,
                        f"{name.replace('/', '_')}_{i:05d}.png"
                    )

                    save_image(frame, save_path)

                print(f"[SAVED] {data.shape[0]} images -> {SAVE_DIR}")

        f.visititems(recursive_search)


if __name__ == "__main__":
    save_images_from_hdf5(HDF5_PATH)