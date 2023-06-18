# pdb 파일 검증
import os
import subprocess
import concurrent.futures

complex_dir = "complex"  # 본인의 디렉터리 경로에 맞게 수정
output_file = "output.txt"  # 출력 파일명

def validate_pdb_file(pdb_file):
    try:
        subprocess.check_output(["pdb_tidy", pdb_file], stderr=subprocess.STDOUT)
        return f"{pdb_file}: Valid PDB file"
    except subprocess.CalledProcessError as e:
        error_message = e.output.decode("utf-8")
        return f"{pdb_file}: Invalid PDB file\nError message: {error_message}"

# complex 디렉터리 내의 모든 파일에 대한 검사를 병렬로 수행하고 결과를 output.txt에 작성
with open(output_file, "w") as output:
    pdb_files = [os.path.join(complex_dir, filename) for filename in os.listdir(complex_dir) if filename.endswith(".pdb")]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(validate_pdb_file, pdb_files)

        for result in results:
            output.write(result + "\n")

print("Validation results written to output.txt")
