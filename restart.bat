@echo off
echo Restarting nanobot container...
docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart
docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f
