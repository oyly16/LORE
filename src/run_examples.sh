python demo_lore.py --resize -1 \
                    --source_prompt "a young woman" \
                    --target_prompt "a young woman with a necklace" \
                    --target_object "necklace" \
                    --target_index 5 \
                    --source_img_dir 'examples/woman.png' \
                    --source_mask_dir 'examples/woman_mask.png'  \
                    --seeds 3 \
                    --savename 'woman'

python demo_lore.py --resize -1 \
                    --source_prompt "a taxi in a neon-lit street" \
                    --target_prompt "a race car in a neon-lit street" \
                    --target_object "race car" \
                    --target_index 1 \
                    --source_img_dir 'examples/car.png' \
                    --source_mask_dir 'examples/car_mask.png'  \
                    --num_steps 30 \
                    --inject 24 \
                    --noise_scale 0.1 \
                    --training_epochs 5 \
                    --seeds 2388791121 \
                    --savename 'car'

python demo_lore.py --resize -1 \
                    --source_prompt "a cup on a wooden table" \
                    --target_prompt "a wooden table" \
                    --target_object "table" \
                    --target_index 2 \
                    --source_img_dir 'examples/cup.png' \
                    --source_mask_dir 'examples/cup_mask.png'  \
                    --num_steps 10 \
                    --inject 8 \
                    --noise_scale 0 \
                    --training_epochs 2 \
                    --seeds 0 \
                    --savename 'cup'