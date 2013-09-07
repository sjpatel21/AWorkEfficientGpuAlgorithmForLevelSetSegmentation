#ifndef CUDA_CONFIG_HPP
#define CUDA_CONFIG_HPP

//
// Use CUDA 3.0 (experimental)
//
//#define CUDA_30

//
// uncomment to disable user selected seed region.  when uncommented,
// the seed region is exactly the same every time.
//
//#define REPRODUCIBLE_SEED

//
// uncomment to record times and compute stats for various components of the algorithm
//
//#define COMPUTE_PERFORMANCE_METRICS
//#define COMPUTE_ACCURACY_METRICS

//
// uncomment to load different ground truth volumes
//
//#define GROUND_TRUTH_WHITE_MATTER
//#define GROUND_TRUTH_WHITE_MATTER_AND_GREY_MATTER

//
// uncomment to use temporally coherent voxel culling algorithm.
// otherwise spatial gradient culling is used only.
//
#define TEMPORAL_DERIVATIVE_VOXEL_CULLING


//
// uncomment to use 6-connected voxel output instead of 27-connected
// voxel output.  slight differences in segmentation occur but significantly
// faster.  we don't currently know which is more accurate and it is very difficult
// to tell by inspection.
//
#define SIX_CONNECTED_VOXEL_CULLING

//
// uncomment to unsafely skip coordinate boundary checking
//
//#define SKIP_BOUNDARY_CHECK_COORDINATES

//
// minimum number of iterations before segmentation terminates
//
#define MINIMUM_SEGMENTATION_ITERATIONS 500;

//
// specify the thresholds below which two voxels are considered equal
//
#define TEMPORAL_DERIVATIVE_THRESHOLD 0.008f
#define SPATIAL_DERIVATIVE_THRESHOLD  0.001f

//
// level set rescaling term
//
#define LEVEL_SET_RESCALE_AMOUNT 0.01f

//
// level set smooth fade for initialization
//
#define LEVEL_SET_INITIALIZE_SMOOTH_DISTANCE 4.0f;


//
// specify the internal representation of level set field values
//
#define LEVEL_SET_CHAR
//#define LEVEL_SET_SHORT

#ifdef LEVEL_SET_CHAR
    #define LEVEL_SET_FIELD_FIXED_POINT
    #define LEVEL_SET_FIELD_TYPE             signed char
    #define LEVEL_SET_FIELD_MAX_VALUE        127.0f
    #define LEVEL_SET_FIELD_EXPORT_MAX_VALUE 127.0f
#endif

#ifdef LEVEL_SET_SHORT
    #define LEVEL_SET_FIELD_FIXED_POINT
    #define LEVEL_SET_FIELD_TYPE             signed short
    #define LEVEL_SET_FIELD_MAX_VALUE        32766.0f
    #define LEVEL_SET_FIELD_EXPORT_MAX_VALUE 127.0f
#endif

//
// specify the tile size
//
#define TILE_SIZE               ( 16 )
#define TILE_SIZE_FLOAT         ( 16.0f )
#define INV_TILE_SIZE           ( 1.0f / TILE_SIZE_FLOAT )
#define TILE_NUM_ELEMENTS       ( TILE_SIZE * TILE_SIZE * TILE_SIZE )
#define TILE_NUM_ELEMENTS_FLOAT ( TILE_SIZE_FLOAT * TILE_SIZE_FLOAT * TILE_SIZE_FLOAT )
#define INV_TILE_NUM_ELEMENTS   ( 1.0f / TILE_NUM_ELEMENTS_FLOAT )

//
// specify the voxel alignment of the loaded volume along the (x, y, z) directions
//
#define VOLUME_ALIGNMENT_X 16
#define VOLUME_ALIGNMENT_Y 16
#define VOLUME_ALIGNMENT_Z 16

//
// anonymous status bar
//
#define ANONYMOUS_STATUS_BAR

//
// white background (remember to change RaycastVolume.cg also)
//
#define WHITE_BACKGROUND

//
// print stuff
//
#define PRINT_USER_INTERACTION_TIME
#define PRINT_NUM_SEGMENTED_VOXELS
//#define PRINT_SIMULATION_TIME
//#define PRINT_NUM_ACTIVE_VOXELS

//
// Uncomment to do lefohn tests.
//
//#define LEFOHN_BENCHMARK

//
// Uncomment to do lefohn tests from file with no debugging.
//
//#define LEFOHN_NO_DEBUG_BENCHMARK

#endif
