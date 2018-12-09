import os
import subprocess


exePath = "../build/LowPoly"
dataDir = "../data"
outDir = "../result"


def testGPU():
	print("Test on GPU")
	logPath = os.path.join(outDir, "timeGPU.log")
	f = open(logPath, "w")
	datas = os.listdir(dataDir)
	datas = sorted(datas, key = lambda x: int(x.split(".")[0]))
	for data in datas:
		print(data)
		dataPath = os.path.join(dataDir, data)
		cmd = [exePath, "-i", dataPath]
		subprocess.call(cmd, stdout=f)
		outImgPath = os.path.join(outDir, data.split(".")[0]+"GPU.png")
		subprocess.call(["mv", "triangle.png", outImgPath])
	f.close()


def testCPU():
	print("Test on CPU")
	logPath = os.path.join(outDir, "timeCPU.log")
	f = open(logPath, "w")
	datas = os.listdir(dataDir)
	datas = sorted(datas, key = lambda x: int(x.split(".")[0]))
	for data in datas:
		print(data)
		dataPath = os.path.join(dataDir, data)
		cmd = [exePath, "-c", "-i", dataPath]
		subprocess.call(cmd, stdout=f)
		outImgPath = os.path.join(outDir, data.split(".")[0]+"CPU.png")
		subprocess.call(["mv", "triangle.png", outImgPath])
	f.close()


if __name__ == "__main__":
	testGPU()
	testCPU()