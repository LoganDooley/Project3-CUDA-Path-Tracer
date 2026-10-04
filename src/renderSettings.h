#pragma once

enum class BvhHeatmapMode : int {
	Total = 0,
	Blas = 1,
	Tlas = 2
};

struct RenderSettings {
	bool bMSAAEnabled = true;
	bool bSVGFEnabled = false;

	bool bStreamCompactionEnabled = false;
	bool bSortPathsByMaterial = false;

	bool bShowBvhHeatmap = false;
	BvhHeatmapMode bvhHeatmapMode = BvhHeatmapMode::Total;
	int bvhHeatmapMaxSteps = 100; // Limit number of steps so anything greater is red
};
