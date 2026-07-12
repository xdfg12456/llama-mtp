from utils.tokenizer import Tokenizer
import os
import io
import glob
import json
import random
import zstandard as zstd
from datasets import Dataset, Features, Sequence, Value

CACHE_DIR = "/workspace/hf_cache"

# 這裡放你的 .jsonl.zst 檔所在資料夾
zstd_download_path = "/workspace/hf_cache/downloads/extracted/94f1af0a24a263bf57426d342697d56ca6e82eb083b011098fccc5505e3af1ba"

TEXT_COLUMN = "text"
NUM_TEST_FILES = 5
SHUFFLE_SEED = 42

tokenizer = Tokenizer(
    model_path="/workspace/app/tokenizer.model"
)


def get_zst_file_list(data_dir):
    zst_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl.zst")))
    if len(zst_files) == 0:
        raise FileNotFoundError(f"No .jsonl.zst files found in: {data_dir}")
    return zst_files


def split_train_test_files(all_files, num_test_files=5, shuffle=False, seed=42):
    if len(all_files) <= num_test_files:
        raise ValueError(
            f"Not enough zst files for split. total={len(all_files)}, num_test_files={num_test_files}"
        )

    files = all_files.copy()
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(files)

    test_files = files[:num_test_files]
    train_files = files[num_test_files:]
    return train_files, test_files


def chunk_file_list(file_list, shard_file_count):
    for i in range(0, len(file_list), shard_file_count):
        yield file_list[i:i + shard_file_count]


def iter_texts_from_zst_files(file_list, text_column=TEXT_COLUMN):
    for fp in file_list:
        print(f"Reading: {fp}")
        with open(fp, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_stream:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = obj.get(text_column, None)
                    if not text:
                        continue

                    yield text


def pack_tokens_from_texts(file_list, max_seq_length=4096):
    """
    逐筆讀 text -> tokenize -> 串接 -> 切成固定長度 chunk
    只存 input_ids，不存 labels
    """
    buffer = []

    for text in iter_texts_from_zst_files(file_list):
        # bos=False, eos=True
        token_ids = tokenizer.encode(text, False, True)

        if token_ids is None or len(token_ids) == 0:
            continue

        buffer.extend(token_ids)

        while len(buffer) >= max_seq_length:
            chunk = buffer[:max_seq_length]
            buffer = buffer[max_seq_length:]

            yield {
                "input_ids": chunk,
            }

    # 最後不足 max_seq_length 的殘段直接丟掉
    # 如果你想保留尾巴，可再另外處理


def build_arrow_dataset_from_files(file_list, max_seq_length=4096):
    features = Features({
        "input_ids": Sequence(Value("int32")),
    })

    ds = Dataset.from_generator(
        lambda: pack_tokens_from_texts(
            file_list,
            max_seq_length=max_seq_length,
        ),
        features=features,
    )
    return ds


def save_sharded_arrow_datasets(
    max_seq_length=4096,
    num_test_files=NUM_TEST_FILES,
    shard_file_count=20,
    output_dir="/workspace/openwebtext2_arrow",
    shuffle=False,
    seed=SHUFFLE_SEED,
):
    all_files = get_zst_file_list(zstd_download_path)

    train_files, test_files = split_train_test_files(
        all_files,
        num_test_files=num_test_files,
        shuffle=shuffle,
        seed=seed,
    )

    train_root = os.path.join(output_dir, "train_shards")
    test_root = os.path.join(output_dir, "test_shards")

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    print(f"Total files       : {len(all_files)}")
    print(f"Train files       : {len(train_files)}")
    print(f"Test files        : {len(test_files)}")
    print(f"Shard file count  : {shard_file_count}")
    print(f"Max seq length    : {max_seq_length}")
    print(f"Output dir        : {output_dir}")

    # train shards
    for shard_idx, shard_files in enumerate(chunk_file_list(train_files, shard_file_count)):
        shard_path = os.path.join(train_root, f"shard_{shard_idx:05d}")

        if os.path.exists(shard_path):
            print(f"[Skip] Train shard already exists: {shard_path}")
            continue

        print(f"[Train] Building shard {shard_idx} with {len(shard_files)} files")
        ds = build_arrow_dataset_from_files(
            shard_files,
            max_seq_length=max_seq_length,
        )

        print(f"[Train] Saving shard {shard_idx} -> {shard_path}")
        ds.save_to_disk(shard_path)

    # test shards
    for shard_idx, shard_files in enumerate(chunk_file_list(test_files, shard_file_count)):
        shard_path = os.path.join(test_root, f"shard_{shard_idx:05d}")

        if os.path.exists(shard_path):
            print(f"[Skip] Test shard already exists: {shard_path}")
            continue

        print(f"[Test] Building shard {shard_idx} with {len(shard_files)} files")
        ds = build_arrow_dataset_from_files(
            shard_files,
            max_seq_length=max_seq_length,
        )

        print(f"[Test] Saving shard {shard_idx} -> {shard_path}")
        ds.save_to_disk(shard_path)

    print("All shards done.")


if __name__ == "__main__":
    save_sharded_arrow_datasets(
        max_seq_length=1024,
        num_test_files=5,
        shard_file_count=5,   # 每 20 個 zst 檔做一個 shard
        output_dir="/workspace/hf_cache/segyges___open_web_text2/default/0.0.0",
        shuffle=True,
        seed=42,
    )