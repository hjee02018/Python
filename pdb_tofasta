import os

# 현재 디렉터리 내의 모든 pdb 파일 가져오기
pdb_files = [f for f in os.listdir('.') if f.endswith('.pdb')]

for pdb_file in pdb_files:
    # pdb 파일 이름에서 pdb id 추출
    pdb_id = pdb_file.split('.')[0]

    # pdb 파일 이름에서 chain id 추출
    chain = pdb_file.split('.')[1]

    # fasta 파일 이름 생성
    fasta_file = f"{pdb_id}.{chain}.fasta"

    # fasta 파일에 header 작성
    header = f">{pdb_id}|{chain}"

    # pdb 파일을 fasta 형식으로 변환하여 fasta 파일에 추가
    os.system(f"pdb_tofasta -multi {pdb_file} | sed '/^>/d' >> {fasta_file}")

    # header를 fasta 파일의 첫 줄에 추가
    with open(fasta_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(header.rstrip('\r\n') + '\n' + content)
