
make_tasks:
	python processes/move_data.py $(audios_path) $(txt_path)
	python processes/cleaning.py clean-transcripts
	python processes/audio_handler.py
	python processes/cleaning.py prepare-for-ls

start_ls:
	POSTGRES_DATA_DIR=~/postgres-data docker-compose up -d

clean:
	rm -f ~/.cache/mfa/*
	rm -f data/done/*
	rm -f data/fail/*
