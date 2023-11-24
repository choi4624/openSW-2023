import asyncio
import datetime
import time
import pytz
import schedule
import telegram

# Define the delay between each call to the job() function
DELAY = 1800.0000



async def job(delay):
    await asyncio.sleep(delay)
    print("my_function() called after {} seconds".format(delay))
    # Update the now variable
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    # Print the current time
    print("current time = ", str(now))

    # Check if the current hour is between 23 and 6
    if now.hour >= 23 or now.hour <= 6:
        return

    # Get the bot token and chat ID
    with open(".token","r") as token_file:
        token = token_file.read()
    with open(".chat_id","r") as chat_id_file:
        chat_id = chat_id_file.read()
      
    # print(token)
    # print(chat_id)

    # Wait for the message to be sent before returning
    await message(token, chat_id)
    ## due to the loop is "run_until_complete", it can be next loop at after message sent. 
    ## it may be 0.3 ~ 1sec 

async def message(token, chat_id):
    # Send a message to the user
    await telegram.Bot(token).send_message(chat_id=chat_id, text='다른 메시지')

async def forever():
    while True:
        await job(DELAY)
        
def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(forever())

if __name__ == '__main__':
  main()