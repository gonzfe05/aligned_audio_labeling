
make_tasks:
	python processes/move_data.py '$(audios_path)' '$(txt_path)'
	python processes/cleaning.py clean-transcripts
	python processes/audio_handler.py
	python processes/cleaning.py prepare-for-ls
	bash ./processes/serve_local_files.sh ~/.cache/segments

start_ls:
	label-studio start -d --log-level=DEBUG
