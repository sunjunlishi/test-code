#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h>
#define MKDIR(dir) mkdir(dir, 0755)
#endif

// 分割文件
bool splitFile(const char* inputFile, const char* outputDir, size_t chunkSize) {
	FILE* fin = fopen(inputFile, "rb");
	if (!fin) {
		printf("Error: Cannot open input file: %s\n", inputFile);
		return false;
	}

	// 创建输出目录
	MKDIR(outputDir);

	// 获取文件名（不含路径）
	std::string inputName = inputFile;
	size_t pos = inputName.find_last_of("/\\");
	if (pos != std::string::npos) {
		inputName = inputName.substr(pos + 1);
	}

	std::vector<char> buffer(chunkSize);
	size_t partNum = 0;
	size_t bytesRead;

	while ((bytesRead = fread(buffer.data(), 1, chunkSize, fin)) > 0) {
		std::string outPath = std::string(outputDir) + "/" + inputName + ".part" + std::to_string(partNum);
		FILE* fout = fopen(outPath.c_str(), "wb");
		if (!fout) {
			printf("Error: Cannot create output file: %s\n", outPath.c_str());
			fclose(fin);
			return false;
		}

		fwrite(buffer.data(), 1, bytesRead, fout);
		fclose(fout);

		printf("Created: %s (%zu bytes)\n", outPath.c_str(), bytesRead);
		partNum++;
	}

	fclose(fin);
	printf("Split complete! Total %zu parts\n", partNum);
	return true;
}

// 合并文件
bool mergeFiles(const std::vector<std::string>& inputFiles, const char* outputFile) {
	FILE* fout = fopen(outputFile, "wb");
	if (!fout) {
		printf("Error: Cannot create output file: %s\n", outputFile);
		return false;
	}

	const size_t bufferSize = 1024 * 1024; // 1MB buffer
	std::vector<char> buffer(bufferSize);

	for (const auto& inputFile : inputFiles) {
		FILE* fin = fopen(inputFile.c_str(), "rb");
		if (!fin) {
			printf("Error: Cannot open input file: %s\n", inputFile.c_str());
			fclose(fout);
			return false;
		}

		size_t bytesRead;
		size_t totalBytes = 0;
		while ((bytesRead = fread(buffer.data(), 1, bufferSize, fin)) > 0) {
			fwrite(buffer.data(), 1, bytesRead, fout);
			totalBytes += bytesRead;
		}

		fclose(fin);
		printf("Merged: %s (%zu bytes)\n", inputFile.c_str(), totalBytes);
	}

	fclose(fout);
	printf("Merge complete! Output: %s\n", outputFile);
	return true;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		printf("Usage:\n");
		printf("  Split: %s split <input.mp4> <output_dir> [chunk_size_mb]\n", argv[0]);
		printf("  Merge: %s merge <output.mp4> <part1> <part2> ...\n", argv[0]);
		printf("\nExample:\n");
		printf("  %s split bin/input.MP4 splits 95\n", argv[0]);
		printf("  %s merge output.MP4 splits/input.MP4.part0 splits/input.MP4.part1 splits/input.MP4.part2\n", argv[0]);
		return 1;
	}

	std::string mode = argv[1];

	if (mode == "split") {
		if (argc < 4) {
			printf("Error: split mode needs input file and output directory\n");
			return 1;
		}

		const char* inputFile = argv[2];
		const char* outputDir = argv[3];
		size_t chunkSize = 95 * 1024 * 1024; // default 95MB

		if (argc >= 5) {
			chunkSize = atoi(argv[4]) * 1024 * 1024;
		}

		printf("Splitting %s into %s (chunk size: %zu MB)\n", inputFile, outputDir, chunkSize / 1024 / 1024);

		if (!splitFile(inputFile, outputDir, chunkSize)) {
			return 1;
		}

	}
	else if (mode == "merge") {
		if (argc < 4) {
			printf("Error: merge mode needs output file and at least one input part\n");
			return 1;
		}

		const char* outputFile = argv[2];
		std::vector<std::string> inputFiles;

		for (int i = 3; i < argc; i++) {
			inputFiles.push_back(argv[i]);
		}

		printf("Merging %zu parts into %s\n", inputFiles.size(), outputFile);

		if (!mergeFiles(inputFiles, outputFile)) {
			return 1;
		}

	}
	else {
		printf("Error: Unknown mode '%s'. Use 'split' or 'merge'\n", mode.c_str());
		return 1;
	}

	return 0;
}