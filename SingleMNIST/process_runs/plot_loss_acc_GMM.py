import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

# Configuration
INPUT_ROOT_FOLDER = "../experiments/csv/GMM"
OUTPUT_ROOT_FOLDER = "plots/loss_acc_plots/GMM"

# Style Configuration
FONT_SIZE_TITLE = 16
FONT_SIZE_AXIS = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 11  # Slightly smaller to fit nicely at bottom

def create_combined_plot(df, output_path, stats):
    """
    Generates a single combined plot with Loss (Left Axis) and Accuracy (Right Axis).
    Legend is placed at the BOTTOM, spanning the width.
    """
    # Standard figure size
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Create a secondary y-axis for Accuracy
    ax2 = ax1.twinx()

    # --- Data Unpacking ---
    ep_min_loss = stats['best_loss_row']['epoch']
    loss_at_min_loss = stats['best_loss_row']['val_loss']
    acc_at_min_loss = stats['best_loss_row']['val_acc']
    
    ep_max_acc = stats['best_acc_row']['epoch']
    loss_at_max_acc = stats['best_acc_row']['val_loss']
    acc_at_max_acc = stats['best_acc_row']['val_acc']

    # --- Plotting ---

    # 1. Accuracy (Right Axis) - FADED
    l_acc_train, = ax2.plot(df['epoch'], df['train_acc'], color='blue', alpha=0.25, linewidth=2, label='Train Acc (Faded)')
    l_acc_val, = ax2.plot(df['epoch'], df['val_acc'], color='red', alpha=0.25, linewidth=2, label='Val Acc (Faded)')

    # 2. Loss (Left Axis) - OPAQUE
    l_loss_train, = ax1.plot(df['epoch'], df['train_loss'], color='blue', alpha=1.0, linewidth=2, label='Train Loss')
    l_loss_val, = ax1.plot(df['epoch'], df['val_loss'], color='red', alpha=1.0, linewidth=2, label='Val Loss')

    # 3. Vertical Markers
    
    # Red Dotted Line (Epoch of Min Loss)
    label_red = (f"Min Val Loss (Ep {int(ep_min_loss)}): val loss: {loss_at_min_loss:.4f}, val acc: {acc_at_min_loss:.4f}")
    l_min_loss = ax1.axvline(x=ep_min_loss, color='red', linestyle=':', linewidth=2, label=label_red)
    
    # Green Dotted Line (Epoch of Max Acc)
    label_green = (f"Max Val Acc (Ep {int(ep_max_acc)}): val loss: {loss_at_max_acc:.4f}, val acc: {acc_at_max_acc:.4f}")
    l_max_acc = ax1.axvline(x=ep_max_acc, color='green', linestyle=':', linewidth=2, label=label_green)

    # --- Formatting ---

    # Axis 1: Loss
    ax1.set_xlabel("Epoch", fontsize=FONT_SIZE_AXIS)
    ax1.set_ylabel("Loss (Symlog)", color='black', fontweight='bold', fontsize=FONT_SIZE_AXIS)
    ax1.set_yscale('symlog', linthresh=0.1)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    # Axis 2: Accuracy
    ax2.set_ylabel("Accuracy", color='gray', fontweight='bold', fontsize=FONT_SIZE_AXIS)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=FONT_SIZE_TICK)

    ax1.set_xlim(df['epoch'].min(), df['epoch'].max())
    ax1.set_title("Combined Training Metrics", fontsize=FONT_SIZE_TITLE)

    # --- Combined Legend (Bottom) ---
    lines = [l_loss_train, l_loss_val, l_acc_train, l_acc_val, l_min_loss, l_max_acc]
    labels = [l.get_label() for l in lines]
    
    # loc='upper center' relative to the anchor point below the axis
    # bbox_to_anchor=(0.5, -0.12) puts it centered horizontally, just below the x-axis
    # ncol=2 splits the items into 2 columns (wider layout)
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12),
               fontsize=FONT_SIZE_LEGEND, framealpha=1.0, 
               ncol=2)

    # bbox_inches='tight' ensures the legend isn't cropped
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
        df = df.sort_values(by="epoch").reset_index(drop=True)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        return

    # Prepare Output Path
    rel_path = os.path.relpath(filepath, INPUT_ROOT_FOLDER) 
    dest_folder = os.path.join(OUTPUT_ROOT_FOLDER, os.path.dirname(rel_path))
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    base_name = os.path.basename(filepath)
    file_root, _ = os.path.splitext(base_name)

    # --- Statistics ---
    idx_min_loss = df['val_loss'].idxmin()
    row_min_loss = df.loc[idx_min_loss]
    
    idx_max_acc = df['val_acc'].idxmax()
    row_max_acc = df.loc[idx_max_acc]
    
    stats = {
        'best_loss_row': row_min_loss,
        'best_acc_row': row_max_acc
    }

    # Generate Combined Plot
    filename = f"plot_combined_{file_root}.png"
    create_combined_plot(
        df, 
        os.path.join(dest_folder, filename), 
        stats
    )

def main():
    # 1. Clean up old plots
    if os.path.exists(OUTPUT_ROOT_FOLDER):
        shutil.rmtree(OUTPUT_ROOT_FOLDER)
        print(f"Deleted old '{OUTPUT_ROOT_FOLDER}' folder.")

    if not os.path.exists(INPUT_ROOT_FOLDER):
        print(f"Error: Folder '{INPUT_ROOT_FOLDER}' does not exist.")
        return

    # 2. Process files
    for root, dirs, files in os.walk(INPUT_ROOT_FOLDER):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                process_file(full_path)

if __name__ == "__main__":
    main()