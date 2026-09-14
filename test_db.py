import asyncio
import asyncpg

async def test():
    try:
        conn = await asyncpg.connect(
            user='admin',
            password='admin123',
            database='mydb',
            host='192.168.86.134',
            port=5432,
            timeout=5
        )
        v = await conn.fetchval('SELECT version()')
        print('SUCCESS:', v[:60])
        
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        print('Tables:', [t['tablename'] for t in tables])
        
        await conn.close()
    except Exception as e:
        print('Failed:', e)

asyncio.run(test())
