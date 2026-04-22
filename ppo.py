import argparse
from logging import config
import os
import sys
import time
import random

import yaml

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logz.batch_logging import batch_log, create_log_dict
from models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
)
from models.icm import ICMEncoder, ICMForward, ICMInverse
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)

from rtpt import RTPT
from craftax.craftax.constants import BlockType, ItemType, MobType

# Code adapted from the original implementation made by Chris Lu
# For this thesis this Code is further modified and extended by Tobias Ulmer
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray

RELEVANT_BLOCKS = jnp.array([
    BlockType.WATER.value,
    BlockType.TREE.value,
    BlockType.FIRE_TREE.value,
    BlockType.COAL.value,
    BlockType.IRON.value,
    BlockType.DIAMOND.value,
    BlockType.SAPPHIRE.value,
    BlockType.RUBY.value,
    BlockType.PLANT.value,
    BlockType.RIPE_PLANT.value,
    BlockType.LAVA.value,
    BlockType.STONE.value,
    BlockType.CRAFTING_TABLE.value,
    BlockType.FURNACE.value,
    BlockType.WALL.value,
    BlockType.WALL_MOSS.value,
    BlockType.STALAGMITE.value,
    BlockType.CHEST.value,
    BlockType.FOUNTAIN.value,
    BlockType.ENCHANTMENT_TABLE_ICE.value,
    BlockType.ENCHANTMENT_TABLE_FIRE.value,
    BlockType.GRAVE.value,
    BlockType.GRAVE2.value,
    BlockType.GRAVE3.value,
    BlockType.NECROMANCER.value,
    # optional:
    # BlockType.LAVA.value,
    # BlockType.CHEST.value,
    # BlockType.FOUNTAIN.value,
], dtype=jnp.int32)

RELEVANT_ITEMS = jnp.array([
    ItemType.TORCH.value,
    ItemType.LADDER_DOWN.value,
    ItemType.LADDER_UP.value,
    ItemType.LADDER_DOWN_BLOCKED.value,
], dtype=jnp.int32)

RESOURCE_BLOCKS = jnp.array([
    BlockType.TREE.value,
    BlockType.FIRE_TREE.value,
    BlockType.COAL.value,
    BlockType.IRON.value,
    BlockType.DIAMOND.value,
    BlockType.SAPPHIRE.value,
    BlockType.RUBY.value,
    BlockType.PLANT.value,
    BlockType.RIPE_PLANT.value,
    BlockType.CHEST.value,
], dtype=jnp.int32)

SOLID_BLOCKS = jnp.array([
    BlockType.STONE.value,
    BlockType.WALL.value,
    BlockType.WALL_MOSS.value,
    BlockType.STALAGMITE.value,
    BlockType.CRAFTING_TABLE.value,
    BlockType.FURNACE.value,
    BlockType.ENCHANTMENT_TABLE_ICE.value,
    BlockType.ENCHANTMENT_TABLE_FIRE.value,
    BlockType.GRAVE.value,
    BlockType.GRAVE2.value,
    BlockType.GRAVE3.value,
    BlockType.NECROMANCER.value,
    BlockType.WATER.value,
], dtype=jnp.int32)

ITEM_CLASSES = jnp.array([
    ItemType.TORCH.value,
    ItemType.LADDER_DOWN.value,
    ItemType.LADDER_UP.value,
    ItemType.LADDER_DOWN_BLOCKED.value,
], dtype=jnp.int32)

DANGER_BLOCKS = jnp.array([
    BlockType.LAVA.value,
], dtype=jnp.int32)

CLASS_PALETTE = jnp.array([
    [0.0, 0.0, 1.0],  # SELF   = blau
    [1.0, 1.0, 0.0],  # RESOURCE = gelb
    [1.0, 1.0, 1.0],  # ITEM   = weiß
    [1.0, 0.0, 0.0],  # DANGER = rot
    [0.0, 1.0, 1.0],  # SOLID  = lila
    [0.0, 0.0, 0.0],  # OTHER  = schwarz
], dtype=jnp.float32)

CKPT_MANAGER = None

CLASS_SELF = 0
CLASS_RESOURCE = 1
CLASS_ITEM = 2
CLASS_DANGER = 3
CLASS_SOLID = 4
CLASS_OTHER = 5

NUM_CLASSES = 6

rtpt = None
UI_HEIGHT = 40
OBS_H = 9
OBS_W = 11
TILE_PX = 10

UI_STATS_ORDER = ["player_health", "player_food", "player_drink", "player_energy", "player_mana"]

INV_SCALARS_ORDER = [
    "wood", "stone", "coal", "iron", "diamond", "ruby", "sapphire",
    "torches", "arrows", "bow", "pickaxe", "sword", "sapling", "armour",
]

def save_gray_inventory_debug_png(config, mask_mode, env, env_params, out_path="gray_inventory_debug.png"):

    """
    Saves the grayscale inventory layer (channel last) for the current mask_mode.
    Only runs once at startup.
    """
    if mask_mode != "occam_inventory":
        return

    import jax
    import jax.numpy as jnp
    import numpy as np
    import imageio.v2 as imageio

    rng = jax.random.PRNGKey(0)
    num_envs = int(config.get("NUM_ENVS", 1))  
    rngs = jax.random.split(rng, num_envs)

    # batched reset: obs (N,H,W,C), state batched
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(rngs, env_params)

    # build masked obs
    masked = mask_selection(obs, state, mask_mode)  # (N,H,W,C)
    # take first env in batch (if N>1)
    x = masked[0] if masked.ndim == 4 else masked

    # grayscale inventory layer
    gray = x[..., -1]  # (H,W)
    gray_np = np.array(gray)

    # normalize to 0..255 for png
    img = (np.clip(gray_np, 0.0, 1.0) * 255.0).astype(np.uint8)

    # save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.imwrite(out_path, img)
    print(f"[debug] saved gray inventory layer to: {out_path}")

def _as_batch(x, N):
    """Make x batched (N, ...) from scalar / (...) / (N,...)"""
    x = jnp.asarray(x)
    if x.ndim == 0:
        return jnp.full((N,), x, dtype=x.dtype)
    if x.ndim == 1:
        if x.shape[0] == N:
            return x
        return jnp.broadcast_to(x[None, :], (N, x.shape[0]))
    if x.ndim == 2:
        return x
    raise ValueError(f"Unexpected shape: {x.shape}")

def _infer_batch_size(core):

    for name in UI_STATS_ORDER:
        v = jnp.asarray(getattr(core, name))
        if v.ndim >= 1:
            return v.shape[0]

    inv = core.inventory
    v = jnp.asarray(inv.potions)
    if v.ndim >= 2:
        return v.shape[0]

    return 1

def extract_inventory_vector(core):
    """
    Returns counts vector:
      (L,) for unbatched or (N,L) for batched.
    L = 14 + 4 + 6 = 24? (scalars list has 14) -> 14 + 4 + 6 = 24
    """
    inv = core.inventory
    N = _infer_batch_size(core)

    parts = []

    # scalars -> (N,1)
    for name in INV_SCALARS_ORDER:
        v = jnp.asarray(getattr(inv, name))
        if v.ndim == 0:
            vb = jnp.full((N, 1), v, dtype=v.dtype)
        elif v.ndim == 1:
            if v.shape[0] == N:
                vb = v.reshape((N, 1))
            else:
                vb = jnp.broadcast_to(v[None, :], (N, v.shape[0]))
        elif v.ndim == 2:
            vb = v
        else:
            raise ValueError(f"Unexpected inventory field shape for {name}: {v.shape}")
        parts.append(vb)

    # potions -> (N,4)
    pot = jnp.asarray(inv.potions)
    if pot.ndim == 1:
        pot = jnp.broadcast_to(pot[None, :], (N, pot.shape[0]))
    parts.append(pot.reshape((N, -1)))

    # books -> (N,6)
    books = jnp.asarray(inv.books)
    if books.ndim == 1:
        books = jnp.broadcast_to(books[None, :], (N, books.shape[0]))
    parts.append(books.reshape((N, -1)))

    out = jnp.concatenate(parts, axis=1)  # (N, 14+4+6)
    return out[0] if N == 1 else out

def player_direction_onehot_layer(core, obs_shape, ui_height=UI_HEIGHT):
    """
    returns: (N,H,W,4) float32
    one-hot direction at player tile
    channel order: [UP, DOWN, RIGHT, LEFT]
    """

    N, H, W, _ = obs_shape

    y0 = (OBS_H // 2) * TILE_PX
    x0 = (OBS_W // 2) * TILE_PX
    y1 = y0 + TILE_PX
    x1 = x0 + TILE_PX

    dir_idx = jnp.asarray(core.player_direction).astype(jnp.int32)

    if dir_idx.ndim == 0:
        dir_idx = jnp.full((N,), dir_idx, dtype=jnp.int32)

    # 0=UP,1=RIGHT,2=DOWN,3=LEFT
    onehot_urdl = jax.nn.one_hot(jnp.clip(dir_idx, 0, 3), 4)

    # reorder → [UP, DOWN, RIGHT, LEFT]
    onehot_udrl = onehot_urdl[:, jnp.array([0, 2, 1, 3])]

    out = jnp.zeros((N, H, W, 4), dtype=jnp.float32)

    tile = onehot_udrl[:, None, None, :]
    tile = jnp.broadcast_to(tile, (N, TILE_PX, TILE_PX, 4))

    out = out.at[:, y0:y1, x0:x1, :].set(tile)

    return out

def ui_stats_vector(core):
    """(5,) or (N,5)"""
    N = _infer_batch_size(core)
    parts = []
    for name in UI_STATS_ORDER:
        v = getattr(core, name)
        vb = _as_batch(v, N).astype(jnp.float32).reshape((N, 1))
        parts.append(vb)
    out = jnp.concatenate(parts, axis=1)
    return out[0] if N == 1 else out

def pack_4x11(values):
    """
    values: (L,) or (N,L) float32 in [0,1]
    returns: (4,11) or (N,4,11)
    """
    values = jnp.asarray(values, dtype=jnp.float32)
    if values.ndim == 1:
        L = values.shape[0]
        assert L <= 44
        grid = jnp.zeros((4, 11), dtype=jnp.float32)
        ys = jnp.arange(L) // 11
        xs = jnp.arange(L) % 11
        return grid.at[ys, xs].set(values)
    else:
        N, L = values.shape
        assert L <= 44
        grid = jnp.zeros((N, 4, 11), dtype=jnp.float32)
        ys = jnp.arange(L) // 11
        xs = jnp.arange(L) % 11
        return grid.at[jnp.arange(N)[:, None], ys[None, :], xs[None, :]].set(values)

def ui_gray_layer_from_state(core, obs, stat_cap=9.0, inv_cap=10.0):
    """
    Builds (N,H,W,1) grayscale HUD/inventory layer.
    Writes a 4x11 tile-grid (encoded values) into the UI pixel region (bottom UI_HEIGHT px).
    """
    obs = jnp.asarray(obs)
    assert obs.ndim == 4, "Pixels env expected obs shape (N,H,W,C)"
    N, H, W = obs.shape[0], obs.shape[1], obs.shape[2]

    # --- stats: (N,5) -> [0,1] with cap=9 ---
    stats = ui_stats_vector(core).astype(jnp.float32)
    if stats.ndim == 1:
        stats = jnp.broadcast_to(stats[None, :], (N, stats.shape[0]))
    stats_gray = jnp.clip(stats, 0.0, stat_cap) / stat_cap  # (N,5)

    # --- inventory: (N,24) -> [0,1] with cap=10 ---
    inv = extract_inventory_vector(core).astype(jnp.float32)
    if inv.ndim == 1:
        inv = jnp.broadcast_to(inv[None, :], (N, inv.shape[0]))
    inv_gray = jnp.clip(inv, 0.0, inv_cap) / inv_cap         # (N,24)

    # --- pack into 4x11 grid (N,4,11) ---
    vec = jnp.concatenate([stats_gray, inv_gray], axis=1)     # (N,29)
    grid4x11 = pack_4x11(vec)                                 # (N,4,11)

    # --- upscale tiles -> pixels: (N,40,110) ---
    grid_px = jnp.repeat(jnp.repeat(grid4x11, TILE_PX, axis=1), TILE_PX, axis=2)

    # --- write into UI region (bottom UI_HEIGHT px), left OBS_W*TILE_PX columns ---
    layer = jnp.zeros((N, H, W), dtype=jnp.float32)
    layer = layer.at[:, H-UI_HEIGHT:H, 0:OBS_W*TILE_PX].set(grid_px)

    return layer[..., None]  # (N,H,W,1)

def zero_ui(x, ui_height=UI_HEIGHT):
    
    if ui_height <= 0:
        return x
    return x.at[:, -ui_height:, :, :].set(0)

def isin_ids(x, ids):
    return (x[..., None] == ids).any(axis=-1)

def onehot_ids_including_zero(view_ids: jnp.ndarray, num_ids: int) -> jnp.ndarray:
    ids = jnp.arange(0, num_ids, dtype=jnp.int32)
    return view_ids[..., None] == ids[None, None, None, :]

def onehot_ids_excluding_zero(view_ids: jnp.ndarray, num_ids: int) -> jnp.ndarray:
    ids = jnp.arange(1, num_ids, dtype=jnp.int32)
    return view_ids[..., None] == ids[None, None, None, :]

def onehot_ids_shifted_by_one(view_ids: jnp.ndarray, num_ids: int) -> jnp.ndarray:
    ids = jnp.arange(1, num_ids + 1, dtype=jnp.int32)
    return view_ids[..., None] == ids[None, None, None, :]

def keep_ui_from_original(masked, original, ui_height):
    if ui_height <= 0:
        return masked
    N, H, W, C = masked.shape
    return masked.at[:, -ui_height:, :, :].set(original[:, -ui_height:, :, :])

def player_tile_rgb_from_obs(obs, ui_height=UI_HEIGHT):
    """
    obs: (N,H,W,C) float32 in [0,1] (Pixels env)
    returns: (N,H,W,3) float32, überall 0 außer Player-Tile (center 10x10)
    """
    N, H, W, C = obs.shape
    game_h = H - ui_height

    # center tile in the 9x11 view
    y0 = (OBS_H // 2) * TILE_PX
    x0 = (OBS_W // 2) * TILE_PX
    y1 = y0 + TILE_PX
    x1 = x0 + TILE_PX

    out = jnp.zeros((N, H, W, 3), dtype=jnp.float32)

    # take RGB from obs 
    if C >= 3:
        tile = obs[:, y0:y1, x0:x1, :3]
    else:
        tile = jnp.broadcast_to(obs[:, y0:y1, x0:x1, :1], (N, TILE_PX, TILE_PX, 3))

    # write only into game area 
    out = out.at[:, y0:y1, x0:x1, :].set(tile)
    return out

def ui_rgb_layer_from_obs(obs, ui_height=UI_HEIGHT):
    """
    obs: (N,H,W,C) float32 [0,1]
    returns: (N,H,W,3) float32, überall 0 außer in der UI-Region (unten ui_height)
    """
    N, H, W, C = obs.shape
    out = jnp.zeros((N, H, W, 3), dtype=jnp.float32)

    if C >= 3:
        ui = obs[:, -ui_height:, :, :3]
    else:
        ui = jnp.broadcast_to(obs[:, -ui_height:, :, :1], (N, ui_height, W, 3))

    out = out.at[:, -ui_height:, :, :].set(ui)
    return out

def view_layers_to_fullframe(view_layers_bool, obs_shape, ui_height=UI_HEIGHT):
    """
    view_layers_bool: (N,9,11,K) bool
    returns: (N,H,W,K) float32 0/1, UI = 0
    """
    N, H, W, _C = obs_shape
    K = view_layers_bool.shape[-1]
    game_h = H - ui_height

    # upsample view (9x11) -> pixel (90x110)
    m = view_layers_bool.astype(jnp.float32)
    m = jnp.repeat(jnp.repeat(m, TILE_PX, axis=1), TILE_PX, axis=2)  # (N,90,110,K)

    full = jnp.zeros((N, H, W, K), dtype=jnp.float32)
    full = full.at[:, :game_h, :OBS_W*TILE_PX, :].set(m)

    return full

def view_mask_to_pixel_mask(view_mask, obs_shape, ui_height=UI_HEIGHT):
    # view_mask: (N,9,11) bool
    N, H, W, C = obs_shape
    game_h = H - ui_height
    game_w = W

    m = view_mask[..., None]  # (N,9,11,1)
    m = jnp.repeat(m, 10, axis=1)  # (N,90,11,1)
    m = jnp.repeat(m, 10, axis=2)  # (N,90,110,1)

    full = jnp.zeros((N, H, W, 1), dtype=bool)
    full = full.at[:, :game_h, :game_w, :].set(m)
    return jnp.broadcast_to(full, (N, H, W, C))

def crop_world_to_view(mask48, player_pos):
    """
    mask48: (N,48,48) bool or int
    player_pos: (N,2) int (y,x)
    returns: (N,9,11) same dtype as mask48
    """
    N = mask48.shape[0]
    pad_y = OBS_H // 2
    pad_x = OBS_W // 2

    
    padded = jnp.pad(mask48, ((0,0), (pad_y, pad_y), (pad_x, pad_x)), mode="constant")

    pos = jnp.asarray(player_pos).astype(jnp.int32)
    py = pos[:, 0] + pad_y
    px = pos[:, 1] + pad_x

    def one(m, y, x):
       
        sy = y - pad_y
        sx = x - pad_x
        return jax.lax.dynamic_slice(m, (sy, sx), (OBS_H, OBS_W))

    return jax.vmap(one)(padded, py, px)

def select_current_level(x, lvl):
    """
    x: (Envs, Level, X, Y)
    lvl: (N,) int in [0, L-1]
    returns: (N, H, W)
    """
    lvl = jnp.asarray(lvl).astype(jnp.int32)
    lvl = jnp.clip(lvl, 0, x.shape[1] - 1)
    
    gathered = jnp.take_along_axis(x, lvl[:, None, None, None], axis=1)  # (N,1,H,W)
    return gathered[:, 0, :, :]

def entity_mask(state):
    lvl = state.player_level

    mobs = select_current_level(
        jnp.asarray(state.mob_map).astype(bool), lvl
    )

    pos = jnp.asarray(state.player_position).astype(jnp.int32)
    N = pos.shape[0]
    py = jnp.clip(pos[:, 0], 0, 47)
    px = jnp.clip(pos[:, 1], 0, 47)

    player = jnp.zeros_like(mobs)
    player = player.at[jnp.arange(N), py, px].set(True)

    return mobs | player

def item_mask(state):
    lvl = jnp.asarray(state.player_level).astype(jnp.int32)
    items = select_current_level(jnp.asarray(state.item_map), lvl)
    return jnp.isin(items, RELEVANT_ITEMS)

def block_mask(state):
    lvl = jnp.asarray(state.player_level).astype(jnp.int32)
    tiles = select_current_level(jnp.asarray(state.map), lvl)
    return jnp.isin(tiles, RELEVANT_BLOCKS)

def mask_binary(obs, env_state):
    
    core = env_state.env_state if hasattr(env_state, "env_state") else env_state

    ent_m = entity_mask(core)
    item_m = item_mask(core)
    block_m = block_mask(core)
    binary_mask = ent_m | item_m | block_m

    view_mask = crop_world_to_view(binary_mask, core.player_position)
    pixel_mask = view_mask_to_pixel_mask(view_mask, obs.shape, ui_height=UI_HEIGHT)
    binary_img = keep_ui_from_original(pixel_mask.astype(obs.dtype), obs, UI_HEIGHT)

    return binary_img

def mask_object(obs, env_state):
    core = env_state.env_state if hasattr(env_state, "env_state") else env_state

    ent_m = entity_mask(core)
    item_m = item_mask(core)
    blk_m = block_mask(core)
    binary_mask = ent_m | item_m | blk_m

    view_mask = crop_world_to_view(binary_mask, core.player_position)
    pixel_mask = view_mask_to_pixel_mask(view_mask, obs.shape, ui_height=UI_HEIGHT)
    masked = jnp.where(pixel_mask, obs, 0.0)
    object_img = keep_ui_from_original(masked, obs, UI_HEIGHT)

    return object_img

def mask_class(obs, env_state):
    core = env_state.env_state if hasattr(env_state, "env_state") else env_state

    ent_m  = entity_mask(core)
    item_m = item_mask(core)
    blk_m  = block_mask(core)
    binary_mask = ent_m | item_m | blk_m

    # current level maps: (N,48,48)
    tiles48 = select_current_level(core.map, core.player_level).astype(jnp.int32)
    items48 = select_current_level(core.item_map, core.player_level).astype(jnp.int32)
    mobs48  = select_current_level(core.mob_map.astype(jnp.int32), core.player_level) > 0  # bool

    # crop to visible view: (N,9,11)
    tiles_v = crop_world_to_view(tiles48, core.player_position)
    items_v = crop_world_to_view(items48, core.player_position)
    mobs_v  = crop_world_to_view(mobs48,  core.player_position)
    bin_v   = crop_world_to_view(binary_mask, core.player_position)

    N = tiles_v.shape[0]

    # 1) Default: OTHER
    class_view = jnp.full((N, OBS_H, OBS_W), CLASS_OTHER, dtype=jnp.int32)

    # 2) SOLID
    class_view = jnp.where(isin_ids(tiles_v, SOLID_BLOCKS), CLASS_SOLID, class_view)

    # 3) RESOURCE
    class_view = jnp.where(isin_ids(tiles_v, RESOURCE_BLOCKS), CLASS_RESOURCE, class_view)

    # 4) ITEM
    class_view = jnp.where(isin_ids(items_v, ITEM_CLASSES), CLASS_ITEM, class_view)

    # 5) DANGER = mobs + lava
    class_view = jnp.where(mobs_v, CLASS_DANGER, class_view)  # <--- HIER kommen die Mobs rein
    class_view = jnp.where(isin_ids(tiles_v, DANGER_BLOCKS), CLASS_DANGER, class_view)
    
    
    # 6) nur relevante Tiles klassifizieren, Rest OTHER
    class_view = jnp.where(bin_v, class_view, CLASS_OTHER)

    # 7) SELF = Player im Zentrum des Views
    class_view = class_view.at[:, OBS_H // 2, OBS_W // 2].set(CLASS_SELF)  # <--- HIER kommt SELF rein


    # auf 90*110 skalieren
    pixel_class_id = jnp.repeat(jnp.repeat(class_view, TILE_PX, axis=1), TILE_PX, axis=2)  # (N,90,110)
    class_rgb = CLASS_PALETTE[pixel_class_id]  # (N,90,110,3)

    N, H, W, C = obs.shape
    game_h = H - UI_HEIGHT

    out = jnp.zeros((N, H, W, 3), dtype=jnp.float32)
    out = out.at[:, :game_h, :OBS_W*TILE_PX, :].set(class_rgb)

    out = keep_ui_from_original(out, obs, UI_HEIGHT)
    return out

def mask_mom(obs, env_state):

    b = mask_binary(obs, env_state)   # (N,H,W,3)
    o = mask_object(obs, env_state)   # (N,H,W,3)
    c = mask_class(obs, env_state)    # (N,H,W,3)

    b = zero_ui(b, UI_HEIGHT)
    o = zero_ui(o, UI_HEIGHT)
    c = zero_ui(c, UI_HEIGHT)
    
    ui_rgb = ui_rgb_layer_from_obs(obs, ui_height=UI_HEIGHT)  # (N,H,W,3)

    # 4) MOM = [binary, object, class, ui_rgb]
    return jnp.concatenate([b, o, c, ui_rgb], axis=-1)

def mask_occam(obs, env_state):
    core = env_state.env_state if hasattr(env_state, "env_state") else env_state

    # current level maps: (N,48,48)
    tiles48 = select_current_level(core.map, core.player_level).astype(jnp.int32)
    items48 = select_current_level(core.item_map, core.player_level).astype(jnp.int32)
    mobs48  = select_current_level(core.mob_map, core.player_level).astype(jnp.int32)

    # crop to view: (N,9,11)
    tiles_v = crop_world_to_view(tiles48, core.player_position)
    items_v = crop_world_to_view(items48, core.player_position)
    mobs_v  = crop_world_to_view(mobs48,  core.player_position)

    N = tiles_v.shape[0]

    # Player center layer: (N,9,11,1)
    player_v = jnp.zeros((N, OBS_H, OBS_W), dtype=bool).at[:, OBS_H // 2, OBS_W // 2].set(True)
    player_layer = player_v[..., None]

    # totals
    num_blocks = len(BlockType)
    num_items  = len(ItemType)
    num_mobs   = len(MobType)

    block_layers = onehot_ids_excluding_zero(tiles_v, num_blocks)   # (N,9,11,num_blocks-1)

    # Items: 
    item_layers  = onehot_ids_excluding_zero(items_v, num_items)    # (N,9,11,num_items-1)

    # Mobs: 
    mob_layers   = onehot_ids_shifted_by_one(mobs_v, num_mobs)      # (N,9,11,num_mobs)

    # Concatenate:
    layers = jnp.concatenate([player_layer, mob_layers, block_layers, item_layers], axis=-1)

    k0 = 0
    PLAYER = (k0, k0 + 1); k0 = PLAYER[1]
    MOBS   = (k0, k0 + mob_layers.shape[-1]); k0 = MOBS[1]
    BLOCKS = (k0, k0 + block_layers.shape[-1]); k0 = BLOCKS[1]
    ITEMS  = (k0, k0 + item_layers.shape[-1]); k0 = ITEMS[1]

    meta = {
        "PLAYER": PLAYER,
        "MOBS": MOBS,
        "BLOCKS": BLOCKS,
        "ITEMS": ITEMS,
        "K": layers.shape[-1],
        "num_blocks": num_blocks,
        "num_items": num_items,
        "num_mobs": num_mobs,
    }

    # 2) Upscale enum layers to full frame, UI stays 0
    layers_full = view_layers_to_fullframe(layers, obs.shape, ui_height=UI_HEIGHT)  # (N,H,W,K) float32

    # 3) Player RGB tile (N,H,W,3), UI=0
    player_rgb = player_tile_rgb_from_obs(obs, ui_height=UI_HEIGHT)

    # 4) Replace player channel (channel 0) with 3 RGB channels.
    #    Keep all other binary enum channels unchanged.
    out = jnp.concatenate([player_rgb, layers_full[..., 1:]], axis=-1)

    # 5) UI RGB layer as extra channels (N,H,W,3)
    ui_rgb = ui_rgb_layer_from_obs(obs, ui_height=UI_HEIGHT)
    out = jnp.concatenate([out, ui_rgb], axis=-1)

    return out

def mask_occam_noPlayerDir(obs, env_state):
    core = env_state.env_state if hasattr(env_state, "env_state") else env_state

    # current level maps: (N,48,48)
    tiles48 = select_current_level(core.map, core.player_level).astype(jnp.int32)
    items48 = select_current_level(core.item_map, core.player_level).astype(jnp.int32)
    mobs48  = select_current_level(core.mob_map, core.player_level).astype(jnp.int32)

    # crop to view: (N,9,11)
    tiles_v = crop_world_to_view(tiles48, core.player_position)
    items_v = crop_world_to_view(items48, core.player_position)
    mobs_v  = crop_world_to_view(mobs48,  core.player_position)

    N = tiles_v.shape[0]

    # Player center layer: (N,9,11,1)
    player_v = jnp.zeros((N, OBS_H, OBS_W), dtype=bool).at[:, OBS_H // 2, OBS_W // 2].set(True)
    player_layer = player_v[..., None]

    # totals
    num_blocks = len(BlockType)
    num_items  = len(ItemType)
    num_mobs   = len(MobType)

    block_layers = onehot_ids_excluding_zero(tiles_v, num_blocks)   # (N,9,11,num_blocks-1)

    # Items: 
    item_layers  = onehot_ids_excluding_zero(items_v, num_items)    # (N,9,11,num_items-1)

    # Mobs: 
    mob_layers   = onehot_ids_shifted_by_one(mobs_v, num_mobs)      # (N,9,11,num_mobs)

    # Concatenate: [PLAYER] + [MOBS] + [BLOCKS] + [ITEMS]
    layers = jnp.concatenate([player_layer, mob_layers, block_layers, item_layers], axis=-1)

    k0 = 0
    PLAYER = (k0, k0 + 1); k0 = PLAYER[1]
    MOBS   = (k0, k0 + mob_layers.shape[-1]); k0 = MOBS[1]
    BLOCKS = (k0, k0 + block_layers.shape[-1]); k0 = BLOCKS[1]
    ITEMS  = (k0, k0 + item_layers.shape[-1]); k0 = ITEMS[1]

    meta = {
        "PLAYER": PLAYER,
        "MOBS": MOBS,
        "BLOCKS": BLOCKS,
        "ITEMS": ITEMS,
        "K": layers.shape[-1],
        "num_blocks": num_blocks,
        "num_items": num_items,
        "num_mobs": num_mobs,
    }

    # 2) Upscale enum layers to full frame, UI stays 0
    layers_full = view_layers_to_fullframe(layers, obs.shape, ui_height=UI_HEIGHT)  # (N,H,W,K) float32

    # 3) Player RGB tile (N,H,W,3), UI=0
    #player_rgb = player_tile_rgb_from_obs(obs, ui_height=UI_HEIGHT)

    # 4) Replace player channel (channel 0) with 3 RGB channels.
    #    Keep all other binary enum channels unchanged.
    #out = jnp.concatenate([player_rgb, layers_full[..., 1:]], axis=-1)

    # 5) UI RGB layer as extra channels (N,H,W,3)
    ui_rgb = ui_rgb_layer_from_obs(obs, ui_height=UI_HEIGHT)
    out = jnp.concatenate([layers_full, ui_rgb], axis=-1)

    return out

def mask_occam_noPlayerDir_noInventory(obs, env_state):
    core = env_state.env_state if hasattr(env_state, "env_state") else env_state

    # current level maps: (N,48,48)
    tiles48 = select_current_level(core.map, core.player_level).astype(jnp.int32)
    items48 = select_current_level(core.item_map, core.player_level).astype(jnp.int32)
    mobs48  = select_current_level(core.mob_map, core.player_level).astype(jnp.int32)

    # crop to view: (N,9,11)
    tiles_v = crop_world_to_view(tiles48, core.player_position)
    items_v = crop_world_to_view(items48, core.player_position)
    mobs_v  = crop_world_to_view(mobs48,  core.player_position)

    N = tiles_v.shape[0]

    # Player center layer: (N,9,11,1)
    player_v = jnp.zeros((N, OBS_H, OBS_W), dtype=bool).at[:, OBS_H // 2, OBS_W // 2].set(True)
    player_layer = player_v[..., None]

    # totals
    num_blocks = len(BlockType)
    num_items  = len(ItemType)
    num_mobs   = len(MobType)

    block_layers = onehot_ids_excluding_zero(tiles_v, num_blocks)   # (N,9,11,num_blocks-1)

    # Items: 
    item_layers  = onehot_ids_excluding_zero(items_v, num_items)    # (N,9,11,num_items-1)

    # Mobs: 
    mob_layers   = onehot_ids_shifted_by_one(mobs_v, num_mobs)      # (N,9,11,num_mobs)

    # Concatenate: [PLAYER] + [MOBS] + [BLOCKS] + [ITEMS]
    layers = jnp.concatenate([player_layer, mob_layers, block_layers, item_layers], axis=-1)

    k0 = 0
    PLAYER = (k0, k0 + 1); k0 = PLAYER[1]
    MOBS   = (k0, k0 + mob_layers.shape[-1]); k0 = MOBS[1]
    BLOCKS = (k0, k0 + block_layers.shape[-1]); k0 = BLOCKS[1]
    ITEMS  = (k0, k0 + item_layers.shape[-1]); k0 = ITEMS[1]

    meta = {
        "PLAYER": PLAYER,
        "MOBS": MOBS,
        "BLOCKS": BLOCKS,
        "ITEMS": ITEMS,
        "K": layers.shape[-1],
        "num_blocks": num_blocks,
        "num_items": num_items,
        "num_mobs": num_mobs,
    }

    # 2) Upscale enum layers to full frame, UI stays 0
    layers_full = view_layers_to_fullframe(layers, obs.shape, ui_height=UI_HEIGHT)  # (N,H,W,K) float32

    return layers_full

def mask_occam_inventory_playerdirection(obs, env_state):

    core = env_state.env_state if hasattr(env_state, "env_state") else env_state

    tiles48 = select_current_level(core.map, core.player_level).astype(jnp.int32)
    items48 = select_current_level(core.item_map, core.player_level).astype(jnp.int32)
    mobs48  = select_current_level(core.mob_map, core.player_level).astype(jnp.int32)

    tiles_v = crop_world_to_view(tiles48, core.player_position)
    items_v = crop_world_to_view(items48, core.player_position)
    mobs_v  = crop_world_to_view(mobs48,  core.player_position)

    N = tiles_v.shape[0]

    player_v = jnp.zeros((N, OBS_H, OBS_W), dtype=bool)\
        .at[:, OBS_H // 2, OBS_W // 2].set(True)
    player_layer = player_v[..., None]

    num_blocks = len(BlockType)
    num_items  = len(ItemType)
    num_mobs   = len(MobType)

    block_layers = onehot_ids_excluding_zero(tiles_v, num_blocks)
    item_layers  = onehot_ids_excluding_zero(items_v, num_items)
    mob_layers   = onehot_ids_shifted_by_one(mobs_v, num_mobs)

    layers = jnp.concatenate(
        [player_layer, mob_layers, block_layers, item_layers],
        axis=-1
    )

    layers_full = view_layers_to_fullframe(
        layers, obs.shape, ui_height=UI_HEIGHT
    )

    # grayscale inventory + stats
    ui_gray = ui_gray_layer_from_state(core, obs, stat_cap=9.0, inv_cap=10.0)

    # direction layer
    dir_layer = player_direction_onehot_layer(core, obs.shape)

    return jnp.concatenate([layers_full, ui_gray, dir_layer], axis=-1)

def mask_occam_noPlayerDir_inventory(obs, env_state):
    core = env_state.env_state if hasattr(env_state, "env_state") else env_state

    # current level maps: (N,48,48)
    tiles48 = select_current_level(core.map, core.player_level).astype(jnp.int32)
    items48 = select_current_level(core.item_map, core.player_level).astype(jnp.int32)
    mobs48  = select_current_level(core.mob_map, core.player_level).astype(jnp.int32)

    # crop to view: (N,9,11)
    tiles_v = crop_world_to_view(tiles48, core.player_position)
    items_v = crop_world_to_view(items48, core.player_position)
    mobs_v  = crop_world_to_view(mobs48,  core.player_position)

    N = tiles_v.shape[0]

    # Player center layer: (N,9,11,1)
    player_v = jnp.zeros((N, OBS_H, OBS_W), dtype=bool).at[:, OBS_H // 2, OBS_W // 2].set(True)
    player_layer = player_v[..., None]

    # totals
    num_blocks = len(BlockType)
    num_items  = len(ItemType)
    num_mobs   = len(MobType)

    block_layers = onehot_ids_excluding_zero(tiles_v, num_blocks)   # (N,9,11,num_blocks-1)

    # Items: 
    item_layers  = onehot_ids_excluding_zero(items_v, num_items)    # (N,9,11,num_items-1)

    # Mobs: 
    mob_layers   = onehot_ids_shifted_by_one(mobs_v, num_mobs)      # (N,9,11,num_mobs)

    # Concatenate: [PLAYER] + [MOBS] + [BLOCKS] + [ITEMS]
    layers = jnp.concatenate([player_layer, mob_layers, block_layers, item_layers], axis=-1)

    k0 = 0
    PLAYER = (k0, k0 + 1); k0 = PLAYER[1]
    MOBS   = (k0, k0 + mob_layers.shape[-1]); k0 = MOBS[1]
    BLOCKS = (k0, k0 + block_layers.shape[-1]); k0 = BLOCKS[1]
    ITEMS  = (k0, k0 + item_layers.shape[-1]); k0 = ITEMS[1]

    meta = {
        "PLAYER": PLAYER,
        "MOBS": MOBS,
        "BLOCKS": BLOCKS,
        "ITEMS": ITEMS,
        "K": layers.shape[-1],
        "num_blocks": num_blocks,
        "num_items": num_items,
        "num_mobs": num_mobs,
    }

    # 2) Upscale enum layers to full frame, UI stays 0
    layers_full = view_layers_to_fullframe(layers, obs.shape, ui_height=UI_HEIGHT)  # (N,H,W,K) float32

    ui_gray = ui_gray_layer_from_state(core, obs, stat_cap=9.0, inv_cap=10.0)  # (N,H,W,1) oder (H,W,1)
    
    out = jnp.concatenate([layers_full, ui_gray], axis=-1)

    return out

def mask_occam_plus(obs, env_state):
    # exakt occam wie bisher
    occam = mask_occam(obs, env_state)

    # Original-Observation als RGB dranhängen
    if obs.shape[-1] >= 3:
        orig_rgb = obs[..., :3]
    else:
        orig_rgb = jnp.broadcast_to(obs[..., :1], obs.shape[:-1] + (3,))

    return jnp.concatenate([occam, orig_rgb], axis=-1)

def mask_occam_plus_noPlayerDir(obs, env_state):
    # exakt occam_noPlayerDir wie bisher
    occam = mask_occam_noPlayerDir(obs, env_state)

    # Original-Observation als RGB dranhängen
    if obs.shape[-1] >= 3:
        orig_rgb = obs[..., :3]
    else:
        orig_rgb = jnp.broadcast_to(obs[..., :1], obs.shape[:-1] + (3,))

    return jnp.concatenate([occam, orig_rgb], axis=-1)

def mask_selection(obs, env_state, mask_mode):
    if mask_mode == "none":
        return obs

    elif mask_mode == "occam":
        return mask_occam(obs, env_state)
    elif mask_mode == "binary":
        return mask_binary(obs, env_state)
    elif mask_mode == "object":
        return mask_object(obs, env_state)
    elif mask_mode == "class":
        return mask_class(obs, env_state)
    elif mask_mode == "mom":
        return mask_mom(obs, env_state)
    elif mask_mode == "occam_plus":
        return mask_occam_plus(obs, env_state)
    elif mask_mode == "occam_noPlayerDir":
        return mask_occam_noPlayerDir(obs, env_state)
    elif mask_mode == "occam_plus_noPlayerDir":
        return mask_occam_plus_noPlayerDir(obs, env_state)
    elif mask_mode == "occam_inventory":  
        return mask_occam_noPlayerDir_inventory(obs, env_state)
    elif mask_mode == "occam_noInventory":  
        return mask_occam_noPlayerDir_noInventory(obs, env_state)
    elif mask_mode == "occam_inventory_playerdirection":  
        return mask_occam_inventory_playerdirection(obs, env_state)

    return obs

def make_train(config):
    #select mask mode
    is_pixels = ("Symbolic" not in config["ENV_NAME"])

    mask_mode = "none"
    if is_pixels:
        if config.get("OCCAM", False):
            mask_mode = "occam"
        elif config.get("MOM", False):
            mask_mode = "mom"
        elif config.get("CLASS", False):
            mask_mode = "class"
        elif config.get("OBJECT", False):
            mask_mode = "object"
        elif config.get("BINARY", False):
            mask_mode = "binary"
        elif config.get("OCCAM_PLUS", False):
            mask_mode = "occam_plus"
        elif config.get("OCCAM_INVENTORY", False):
            mask_mode = "occam_inventory"
        elif config.get("OCCAM_NOPLAYERDIR", False):
            mask_mode = "occam_noPlayerDir"
        elif config.get("OCCAM_PLUS_NOPLAYERDIR", False):
            mask_mode = "occam_plus_noPlayerDir"
        elif config.get("OCCAM_NOINVENTORY", False):
            mask_mode = "occam_noInventory"
        elif config.get("OCCAM_INVENTORY_PLAYERDIRECTION", False):
            mask_mode = "occam_inventory_playerdirection"

    num_updates = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    # NEW: runtime limit
    max_runtime = config.get("MAX_RUNTIME_SECONDS", 0)

    if max_runtime > 0:

        config["NUM_UPDATES"] = num_updates  
    else:
        config["NUM_UPDATES"] = num_updates


    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params
    
    save_gray_inventory_debug_png(config, mask_mode, env, env_params,
                              out_path="analysis/debug/gray_inventory_layer.png")

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
        else:
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )

        # Exploration state
        ex_state = {
            "icm_encoder": None,
            "icm_forward": None,
            "icm_inverse": None,
            "e3b_matrix": None,
        }
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        

        if config["TRAIN_ICM"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # Encoder
            icm_encoder_network = ICMEncoder(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_encoder_network_params = icm_encoder_network.init(
                _rng, jnp.zeros((1, obs_shape))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_encoder"] = TrainState.create(
                apply_fn=icm_encoder_network.apply,
                params=icm_encoder_network_params,
                tx=tx,
            )

            # Forward
            icm_forward_network = ICMForward(
                num_layers=3,
                output_dim=config["ICM_LATENT_SIZE"],
                layer_size=config["ICM_LAYER_SIZE"],
                num_actions=env.num_actions,
            )
            rng, _rng = jax.random.split(rng)
            icm_forward_network_params = icm_forward_network.init(
                _rng, jnp.zeros((1, config["ICM_LATENT_SIZE"])), jnp.zeros((1,))
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_forward"] = TrainState.create(
                apply_fn=icm_forward_network.apply,
                params=icm_forward_network_params,
                tx=tx,
            )

            # Inverse
            icm_inverse_network = ICMInverse(
                num_layers=3,
                output_dim=env.num_actions,
                layer_size=config["ICM_LAYER_SIZE"],
            )
            rng, _rng = jax.random.split(rng)
            icm_inverse_network_params = icm_inverse_network.init(
                _rng,
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
                jnp.zeros((1, config["ICM_LATENT_SIZE"])),
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ICM_LR"], eps=1e-5),
            )
            ex_state["icm_inverse"] = TrainState.create(
                apply_fn=icm_inverse_network.apply,
                params=icm_inverse_network_params,
                tx=tx,
            )

            if config["USE_E3B"]:
                ex_state["e3b_matrix"] = (
                    jnp.repeat(
                        jnp.expand_dims(
                            jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                        ),
                        config["NUM_ENVS"],
                        axis=0,
                    )
                    / config["E3B_LAMBDA"]
                )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv0, env_state0 = env.reset(_rng, env_params)
        obsv0 = mask_selection(obsv0, env_state0, mask_mode)

        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, obsv0[:1])

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        resume_step = config.get("RESUME_CHECKPOINT_STEP", None)

        if resume_step is not None:
            if CKPT_MANAGER is None:
                raise ValueError("Checkpoint manager is not initialized.")

            available_steps = list(CKPT_MANAGER.all_steps())
            if resume_step not in available_steps:
                raise ValueError(
                    f"Requested checkpoint step {resume_step} not found. "
                    f"Available steps: {available_steps}"
                )

            restored = CKPT_MANAGER.restore(resume_step, items=train_state)
            train_state = restored
            print(f"[ckpt] restored checkpoint step={resume_step}")

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    ex_state,
                    rng,
                    update_step,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                obsv = mask_selection(obsv, env_state, mask_mode)

                reward_i = jnp.zeros(config["NUM_ENVS"])

                if config["TRAIN_ICM"]:
                    latent_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, last_obs
                    )
                    latent_next_obs = ex_state["icm_encoder"].apply_fn(
                        ex_state["icm_encoder"].params, obsv
                    )

                    latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                        ex_state["icm_forward"].params, latent_obs, action
                    )
                    error = (latent_next_obs - latent_next_obs_pred) * (
                        1 - done[:, None]
                    )
                    mse = jnp.square(error).mean(axis=-1)

                    reward_i = mse * config["ICM_REWARD_COEFF"]

                    if config["USE_E3B"]:
                        # Embedding is (NUM_ENVS, 128)
                        # e3b_matrix is (NUM_ENVS, 128, 128)
                        us = jax.vmap(jnp.matmul)(ex_state["e3b_matrix"], latent_obs)
                        bs = jax.vmap(jnp.dot)(latent_obs, us)

                        def update_c(c, b, u):
                            return c - (1.0 / (1 + b)) * jnp.outer(u, u)

                        updated_cs = jax.vmap(update_c)(ex_state["e3b_matrix"], bs, us)
                        new_cs = (
                            jnp.repeat(
                                jnp.expand_dims(
                                    jnp.identity(config["ICM_LATENT_SIZE"]), axis=0
                                ),
                                config["NUM_ENVS"],
                                axis=0,
                            )
                            / config["E3B_LAMBDA"]
                        )
                        ex_state["e3b_matrix"] = jnp.where(
                            done[:, None, None], new_cs, updated_cs
                        )

                        e3b_bonus = jnp.where(
                            done, jnp.zeros((config["NUM_ENVS"],)), bs
                        )

                        reward_i = e3b_bonus * config["E3B_REWARD_COEFF"]

                reward = reward_e + reward_i

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=obsv,
                    info=info,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    ex_state,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step,
            ) = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, losses = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]

            save_every = int(config.get("SAVE_EVERY_ENV_STEPS", 0))
            if CKPT_MANAGER is not None and save_every and config.get("USE_WANDB", False):

                # env_steps after this update (same calc wie dein logging)
                env_steps = (update_step + 1) * (config["NUM_ENVS"] * config["NUM_STEPS"])

                def _save_ckpt(ts, step):
                    save_args = orbax_utils.save_args_from_target(ts)
                    CKPT_MANAGER.save(
                        int(step),
                        ts,
                        save_kwargs={"save_args": save_args},
                    )
                    
                    print(f"[ckpt] saved step={int(step)}")

                # nur alle save_every steps
                do_save = (env_steps % save_every) == 0

                
                jax.lax.cond(
                    do_save,
                    lambda _: jax.debug.callback(_save_ckpt, train_state, env_steps),
                    lambda _: None,
                    operand=None,
                )

            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    def _inverse_loss_fn(
                        icm_encoder_params, icm_inverse_params, traj_batch
                    ):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            icm_encoder_params, traj_batch.next_obs
                        )

                        action_pred_logits = ex_state["icm_inverse"].apply_fn(
                            icm_inverse_params, latent_obs, latent_next_obs
                        )
                        true_action = jax.nn.one_hot(
                            traj_batch.action, num_classes=action_pred_logits.shape[-1]
                        )

                        bce = -jnp.mean(
                            jnp.sum(
                                action_pred_logits
                                * true_action
                                * (1 - traj_batch.done[:, None]),
                                axis=1,
                            )
                        )

                        return bce * config["ICM_INVERSE_LOSS_COEF"]

                    inverse_grad_fn = jax.value_and_grad(
                        _inverse_loss_fn,
                        has_aux=False,
                        argnums=(
                            0,
                            1,
                        ),
                    )
                    inverse_loss, grads = inverse_grad_fn(
                        ex_state["icm_encoder"].params,
                        ex_state["icm_inverse"].params,
                        traj_batch,
                    )
                    icm_encoder_grad, icm_inverse_grad = grads
                    ex_state["icm_encoder"] = ex_state["icm_encoder"].apply_gradients(
                        grads=icm_encoder_grad
                    )
                    ex_state["icm_inverse"] = ex_state["icm_inverse"].apply_gradients(
                        grads=icm_inverse_grad
                    )

                    def _forward_loss_fn(icm_forward_params, traj_batch):
                        latent_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.obs
                        )
                        latent_next_obs = ex_state["icm_encoder"].apply_fn(
                            ex_state["icm_encoder"].params, traj_batch.next_obs
                        )

                        latent_next_obs_pred = ex_state["icm_forward"].apply_fn(
                            icm_forward_params, latent_obs, traj_batch.action
                        )

                        error = (latent_next_obs - latent_next_obs_pred) * (
                            1 - traj_batch.done[:, None]
                        )
                        return (
                            jnp.square(error).mean() * config["ICM_FORWARD_LOSS_COEF"]
                        )

                    forward_grad_fn = jax.value_and_grad(
                        _forward_loss_fn, has_aux=False
                    )
                    forward_loss, icm_forward_grad = forward_grad_fn(
                        ex_state["icm_forward"].params, traj_batch
                    )
                    ex_state["icm_forward"] = ex_state["icm_forward"].apply_gradients(
                        grads=icm_forward_grad
                    )

                    losses = (inverse_loss, forward_loss)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                ex_state, losses = jax.lax.scan(
                    _update_ex_minbatch, ex_state, minibatches
                )
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["TRAIN_ICM"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(
                    _update_ex_epoch,
                    ex_update_state,
                    None,
                    config["EXPLORATION_UPDATE_EPOCHS"],
                )
                metric["icm_inverse_loss"] = ex_loss[0].mean()
                metric["icm_forward_loss"] = ex_loss[1].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                ex_state = ex_update_state[0]
                rng = ex_update_state[-1]

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)

                    ep_ret = to_log.get("episode_return", None)
                    if ep_ret is None:
                        ep_ret = to_log.get("reward", 0.0)

                    ep_ret = float(ep_ret)
                    to_log["norm_episode_return"] = (ep_ret / 226.0) * 100.0

                    env_steps = (int(update_step) + 1) * (config["NUM_ENVS"] * config["NUM_STEPS"])
                    to_log["env_steps"] = env_steps
                    to_log["timestep_m"] = env_steps / 1e6


                    batch_log(update_step, to_log, config)

                    if rtpt is not None:
                        rtpt.step()

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )
            
            if config.get("MAX_RUNTIME_SECONDS", 0) > 0:

                def _check_time(update_step):
                    elapsed = time.time() - config["START_TIME"]
                    if elapsed > config["MAX_RUNTIME_SECONDS"]:
                        print(f"[STOP] Max runtime reached ({elapsed:.2f}s)")
                        raise RuntimeError("TIME_LIMIT_REACHED")

                jax.debug.callback(_check_time, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                ex_state,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state0,
            obsv0,
            ex_state,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}  # , "info": metric}


    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}
    config["START_TIME"] = time.time()

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    ckpt_dir = os.path.join("/tmp/ckpts", config["CHECKPOINT_DIR"])
    os.makedirs(ckpt_dir, exist_ok=True)
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(config_path):
        
        with open(config_path, "w") as f:
            def _yamlable(x):
                import numpy as _np
                if isinstance(x, (_np.integer,)):
                    return int(x)
                if isinstance(x, (_np.floating,)):
                    return float(x)
                if isinstance(x, (_np.bool_,)):
                    return bool(x)
                return x
            yaml.safe_dump({k: _yamlable(v) for k, v in config.items()}, f)

        print(f"[ckpt] saved config to {config_path}")
    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=50, create=True)
    checkpoint_manager = CheckpointManager(ckpt_dir, orbax_checkpointer, options)

    global CKPT_MANAGER
    CKPT_MANAGER = checkpoint_manager

    with open(os.path.join(wandb.run.dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)
    print(f"[ckpt] writing to: {ckpt_dir}")

    rtpt = RTPT(
        name_initials="TU",
        experiment_name="Craftax",
        max_iterations=1000000,
    )
    rtpt.start()

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    try:
        out = train_vmap(rngs)
    except RuntimeError as e:
        if "TIME_LIMIT_REACHED" in str(e):
            print("Training stopped due to time limit.")
            out = None
        else:
            raise
    t1 = time.time()

    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--max_runtime_seconds",
    type=int,
    default=0,
    help="Stop training after X seconds (0 = disabled)"
    )
    parser.add_argument("--deterministic", action="store_true",
                    help="Try to make GPU execution deterministic (may be slower).")
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--occam_noInventory",
        action="store_true",
        help="OCCAM without player RGB and without inventory/UI layer."
    )
    parser.add_argument(
        "--occam_inventory_playerdirection",
        action="store_true",
        help=""
    )
    parser.add_argument("--occam_inventory", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="policies_ckpt")
    parser.add_argument("--save_every_env_steps", type=int, default=10_000)
    parser.add_argument(
    "--resume_checkpoint_step",
    type=int,
    default=None,
    help="Load this checkpoint step from /tmp/ckpts/<checkpoint_dir> and continue training from it."
    )
    parser.add_argument("--occam_noPlayerDir", action="store_true", help="Use occam enum layers but keep player as binary (no player RGB direction tile).")
    parser.add_argument("--occam_plus_noPlayerDir", action="store_true", help="Use occam enum layers but keep player as binary (no player RGB direction tile).")
    parser.add_argument("--occam", action="store_true", help="Use enum layers instead of raw pixels (pixels env only).")
    parser.add_argument("--occam_plus", action="store_true", help="Use enum layers + original RGB instead of raw pixels (pixels env only).")
    parser.add_argument("--mom", action="store_true", help="Use plane mask instead of raw pixels (pixels env only).")
    parser.add_argument("--random_policy", action="store_true", help="Ignore network and sample random actions.")
    parser.add_argument("--class", action="store_true", help="Use class mask instead of raw pixels (pixels env only).")
    parser.add_argument("--object", action="store_true", help="Use object mask instead of raw pixels (pixels env only).")
    parser.add_argument("--binary", action="store_true", help="Use binary mask instead of raw pixels (pixels env only).")
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e9
    )  
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ICM
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)

    args, rest_args = parser.parse_known_args(sys.argv[1:])

    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
