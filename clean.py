import json
import re
from pathlib import Path
import tiktoken
import torch
from tqdm.auto import tqdm

# ==========================================
# 1. 설정값 (압축 해제된 폴더 경로로 수정)
# ==========================================
DATA_DIR = Path(r"C:\Users\wirew\Downloads\NIKL_DIALOGUE_2025_v1.0\NIKL_DIALOGUE_2025_v1.0")        # 압축 푼 폴더 경로
OUT_PT_PATH = Path("nikl_qa_dataset.pt")         # 저장할 .pt 경로
CONTEXT_LENGTH = 256                             # MODEL_CONFIG["context_length"]와 동일
STRIDE = 128                                     # 슬라이딩 윈도우 간격
MAX_FILES = None                                 # 테스트할 땐 100 정도로 제한, 전체는 None
ONLY_QUESTIONS = False                           # True면 '?'로 끝나는 발화만 질문으로 추출

# ==========================================
# 2. 텍스트 정제 함수
# ==========================================
def clean_utterance(text: str) -> str:
    # 국립국어원 비식별화/전사 태그 예: &이름&, ((웃음)), {laughing} 등 제거
    text = re.sub(r"&[^&\s]+&", "", text)
    text = re.sub(r"\(\([^)]*\)\)|\{[^}]*\}", "", text)
    # 한글, 영문, 기본 구두점(.,?!), 공백만 남김
    text = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9.,?!\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==========================================
# 3. JSON에서 연속 화자 교대(Q&A) 쌍 추출
# ==========================================
def extract_qa_pairs_from_json(data: dict, only_questions: bool = False) -> list[tuple[str, str]]:
    pairs = []
    docs = data.get("document", [])
    if isinstance(docs, dict):
        docs = [docs]

    for doc in docs:
        utterances = doc.get("utterance", [])
        if not utterances:
            continue

        merged_turns = []
        for utt in utterances:
            speaker = utt.get("speaker_id") or utt.get("speaker") or ""
            raw_text = utt.get("form") or utt.get("text") or ""
            cleaned = clean_utterance(raw_text)
            if not cleaned:
                continue

            if merged_turns and merged_turns[-1][0] == speaker:
                merged_turns[-1] = (speaker, merged_turns[-1][1] + " " + cleaned)
            else:
                merged_turns.append((speaker, cleaned))

        for i in range(len(merged_turns) - 1):
            q_spk, q_text = merged_turns[i]
            a_spk, a_text = merged_turns[i + 1]

            if q_spk == a_spk:
                continue
            if len(q_text) < 2 or len(a_text) < 2:
                continue
            if only_questions and not q_text.endswith("?"):
                continue

            pairs.append((q_text, a_text))

    return pairs

# ==========================================
# 4. 폴더 내 JSON 순회 -> 토큰화 -> .pt 저장
# ==========================================
def build_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {DATA_DIR.resolve()}")

    json_files = sorted(DATA_DIR.rglob("*.json"))
    if MAX_FILES is not None:
        json_files = json_files[:MAX_FILES]

    if not json_files:
        raise FileNotFoundError(f"{DATA_DIR} 안에 .json 파일이 없습니다.")

    enc = tiktoken.get_encoding("o200k_base")
    eot = "<|endoftext|>"

    all_qa_pairs = []
    print(f"총 {len(json_files):,}개 JSON 파일 파싱 시작...")
    for json_path in tqdm(json_files, desc="JSON 파싱"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
            all_qa_pairs.extend(extract_qa_pairs_from_json(content, only_questions=ONLY_QUESTIONS))
        except Exception as e:
            print(f"[경고] {json_path.name} 파싱 실패: {e}")

    print(f"추출된 Q&A 쌍 개수: {len(all_qa_pairs):,}개")
    if not all_qa_pairs:
        raise ValueError("추출된 Q&A 쌍이 없습니다. 샘플 JSON 1개의 키 구조를 확인해 주세요.")

    print("\n[추출 샘플 확인]")
    for q, a in all_qa_pairs[:3]:
        print(f"질문: {q}\n답변: {a}\n" + "-" * 40)

    formatted_texts = [f"질문: {q}\n답변: {a}{eot}" for q, a in all_qa_pairs]
    full_text = "\n".join(formatted_texts)

    print("o200k_base 토큰화 중...")
    token_ids = enc.encode(full_text, allowed_special={eot})
    print(f"전체 토큰 수: {len(token_ids):,}개")

    input_chunks = []
    target_chunks = []
    for i in range(0, len(token_ids) - CONTEXT_LENGTH, STRIDE):
        input_chunks.append(token_ids[i : i + CONTEXT_LENGTH])
        target_chunks.append(token_ids[i + 1 : i + CONTEXT_LENGTH + 1])

    inputs_tensor = torch.tensor(input_chunks, dtype=torch.long)
    targets_tensor = torch.tensor(target_chunks, dtype=torch.long)

    print(f"최종 텐서 shape -> inputs: {inputs_tensor.shape}, targets: {targets_tensor.shape}")

    torch.save((inputs_tensor, targets_tensor), OUT_PT_PATH)
    print(f"저장 완료: {OUT_PT_PATH.resolve()}")

if __name__ == "__main__":
    build_dataset()