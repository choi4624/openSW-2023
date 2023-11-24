import os
import asyncio
import datetime
import time
import pytz
import schedule
import telegram

from openai import OpenAI

 

with open(".key","r") as key_file:
        key = key_file.read()
client = OpenAI(api_key=key)

def openai(chat):
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Response as korean language, even request message is english. way is not limited. It can be enable translate English to Korean"},
        {"role": "user", "content": chat}
    ]
    )
    response_message = completion.choices[0].message
    response_message = response_message.content
        
    return response_message

async def get_message():
    bot = telegram.Bot(token = get_token())
    updates = await bot.get_updates()
    for u in updates:
        print(u.message['chat']['id'])
        print(u.message['text'])
        chat_sentence = u.message['text']
    #chat_id = updates.message.chat.id
    
    return chat_sentence
    
def get_token():
    # Get the bot token and chat ID
    with open(".token","r") as token_file:
        token = token_file.read()    
    return token

def get_chatid():
    with open(".chat_id","r") as chat_id_file:
        chat_id = chat_id_file.read()
    return chat_id

previous_message = None         

async def job():
    # await asyncio.sleep()
    # print("my_function() called after {} seconds".format(delay))
    # Update the now variable
    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
    # Print the current time
    # print("current time = ", str(now))

    token = get_token()
    chat_id = get_chatid()
    # print(token)
    # print(chat_id)
    chat_sentence = await get_message()
    
    ## if the previous_message is same as chat_sentence. skip 
    global previous_message
    if chat_sentence != previous_message:
        print('prev' + previous_message)
        previous_message = chat_sentence
        print('chat' + chat_sentence)
    # Wait for the message to be sent before returning
        await message(token, chat_id,chat_sentence)
    ## due to the loop is "run_until_complete", it can be next loop at after message sent. 
    ## it may be 0.3 ~ 1sec 


async def message(token, chat_id, chat_sentence):
    response_message = openai(chat_sentence)
    
    
    # Send a message to the user
    await telegram.Bot(token).send_message(chat_id=chat_id, text=response_message)


async def forever():     
    while True:
        await job()
        
def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(forever())

if __name__ == '__main__':
  main()

