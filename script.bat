@echo off

set agent=%1
set scenario=%2

python .\game.py --save-dir ./Data/%scenario%/0/%agent%/        --scenario %scenario% --plots-dir ./Data/%scenario%/0/%agent%/      --exp-name %agent% --num-adversaries 3 --good-mic 0  --adv-mic 0 
python .\game.py --save-dir ./Data/%scenario%/1e-1/%agent%/     --scenario %scenario% --plots-dir ./Data/%scenario%/1e-1/%agent%/   --exp-name %agent% --num-adversaries 3 --good-mic 1e-1  --adv-mic 0 
python .\game.py --save-dir ./Data/%scenario%/1e-2/%agent%/     --scenario %scenario% --plots-dir ./Data/%scenario%/1e-2/%agent%/   --exp-name %agent% --num-adversaries 3 --good-mic 1e-2 --adv-mic 0
python .\game.py --save-dir ./Data/%scenario%/1e-3/%agent%/     --scenario %scenario% --plots-dir ./Data/%scenario%/1e-3/%agent%/   --exp-name %agent% --num-adversaries 3 --good-mic 1e-3 --adv-mic 0 
python .\game.py --save-dir ./Data/%scenario%/1e-4/%agent%/     --scenario %scenario% --plots-dir ./Data/%scenario%/1e-4/%agent%/   --exp-name %agent% --num-adversaries 3 --good-mic 1e-4 --adv-mic 0 



