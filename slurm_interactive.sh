salloc --time=2:0:0 --mem-per-cpu=4G --ntasks=1 --cpus-per-task=16 --account=rrg-yuntian --gpus=h100:1
salloc --time=0:30:0 --mem-per-cpu=4G --ntasks=1 --cpus-per-task=16 --account=def-yuntian --gpus=h100:1
salloc --time=1:0:0 --mem-per-cpu=2G --ntasks=1 --cpus-per-task=16 --account=rrg-yuntian
salloc --time=2:0:0  --mem=240G --ntasks=1 --cpus-per-task=16 --account=rrg-yuntian --gpus=h100:2

salloc --time=2:0:0 --mem-per-cpu=4G --ntasks=1 --cpus-per-task=8 --gpus=1

salloc --time=5:00:0 --mem=64G --ntasks=1 --cpus-per-task=16 --account=aip-yuntian --gpus=h100:1
salloc --time=8:00:0 --mem=64G --ntasks=1 --cpus-per-task=16 --account=aip-yuntian --gpus=h100:1