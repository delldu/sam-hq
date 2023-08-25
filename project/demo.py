# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Thu 13 Jul 2023 12:34:35 AM CST
# ***
# ************************************************************************************/
#
import SAM
import torch

T = 15
test_database = {}
test_database["images/example0.png"] = torch.tensor([
		[4 + T, 13 + T, 1007 - T, 1023 - T],
	])
test_database["images/example1.png"] = torch.tensor([
		[306 + T, 132 + T, 925 - T, 893 - T],
	])
test_database["images/example2.png"] = torch.tensor([
		[495, 518, 495, 518],
		[217, 140, 217, 140],
	])
test_database["images/example3.png"] = torch.tensor([
		[221, 482, 221, 482],
		[498, 633, 498, 633],
		[750, 379, 750, 379],	
	])
test_database["images/example4.png"] = torch.tensor([
		[64 + T,76 + T, 940 - T, 919 - T],
	])
test_database["images/example5.png"] = torch.tensor([
		[373, 363, 373, 363],
		[452, 575, 452, 575],
	])
test_database["images/example6.png"] = torch.tensor([
		[181 + T, 196 + T, 757 - T, 495 - T],
	])
test_database["images/example7.png"] = torch.tensor([
		[45 + T,260 + T, 515 - T, 470 - T],
		[310 + T, 228 + T, 424 - T, 296 - T],
	])

SAM.predict(test_database, "output")

