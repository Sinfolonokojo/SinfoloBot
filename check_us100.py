import MetaTrader5 as mt5

mt5.initialize()

print('Checking US100 symbol:')
info = mt5.symbol_info('US100')
if info:
    print(f'  Found: {info.name} - {info.description}')
else:
    print('  Not found, attempting to enable...')
    result = mt5.symbol_select('US100', True)
    print(f'  Enable result: {result}')
    info = mt5.symbol_info('US100')
    if info:
        print(f'  Now found: {info.name} - {info.description}')
    else:
        print('  Still not available - US100 not supported on this server')

mt5.shutdown()
