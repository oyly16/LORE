# optim noise (ours)
python vis_attn_lore.py --resize -1 \
                    --source_prompt "a cat and a dog" \
                    --target_prompt "a cat and a race car" \
                    --target_object "race car" \
                    --target_index 4 \
                    --source_img_dir 'examples/catdog.jpg' \
                    --source_mask_dir 'examples/dog_mask.png'  \
                    --noise_scale 0.1 \
                    --training_epochs 30 \
                    --seeds 25 \
                    --savename 'catdog_ours'

# inversion noise (rf-edit)
python vis_attn_lore.py --resize -1 \
                    --source_prompt "a cat and a dog" \
                    --target_prompt "a cat and a race car" \
                    --target_object "race car" \
                    --target_index 4 \
                    --source_img_dir 'examples/catdog.jpg' \
                    --source_mask_dir 'examples/dog_mask.png'  \
                    --inject 5 \
                    --training_epochs 0 \
                    --noise_scale 0 \
                    --v_inject 0 \
                    --savename 'catdog_inversion'
