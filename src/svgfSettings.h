#pragma once

struct SVGFSettings {
	// Temporal accumulation
	float colorAlpha = 0.2f;
	float momentsAlpha = 0.2f;

	// A-trous edge-stopping
	float sigmaLuminance = 4.0f;
	float sigmaNormal = 128.0f;
	float sigmaDepth = 1.0f;
	int atrousIterations = 5;
};
