# export ISAAC_OBJ_DIR=/home/galois/Downloads/isaac-sim-assets-4.5.0/Assets/Isaac/4.5/Isaac
# export YCB_DIR=/home/galois/assets/assets/obj/YCB/Axis_Aligned_Physics
# export SAVE_DIR=/home/galois/temp
# export ASSETS_DIR=/home/galois/assets/assets

export ISAAC_OBJ_DIR=/home/ikun/assets/Assets/Isaac/4.5/Isaac
export YCB_DIR=/home/ikun/assets/assets/obj/YCB/Axis_Aligned_Physics
export SAVE_DIR=/home/ikun/temp
export ASSETS_DIR=/home/ikun/assets/assets

if [ -n "${BASH_SOURCE:-}" ]; then
    export BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
else
    export BASE_DIR=$PWD
fi
export PATH="$BASE_DIR/bin:$PATH"
