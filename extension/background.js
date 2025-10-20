const PROTOCOL_VERSION = '1.3';
const HIGH_ENTROPY_API_CATEGORY =
  'disabled-by-default-identifiability.high_entropy_api';
const TRACE_CONFIG = {
  'includedCategories': [HIGH_ENTROPY_API_CATEGORY],
  'excludedCategories': ['*'],
};
const TRACING_TIME_MILLISECONDS = 10000;
const PORT = 4000;
const PORT2 = 5000;

/**
 * Use the debugger api to start the tracing of high entropy apis.
 * @param {number} tabId the Tab for which to start the tracing.
 */
async function startTracing(tabId) {
  try {
    const tab = await chrome.tabs.get(tabId);
    if (tab.url.startsWith('chrome://')) {
      // Skip tracing for chrome:// URLs
      console.log('Skipping tracing for chrome:// URL');
      return;
    }
    await chrome.debugger.attach({ tabId }, PROTOCOL_VERSION);
    // https://chromedevtools.github.io/devtools-protocol/tot/Debugger/#method-enable
    await chrome.debugger.sendCommand({ tabId }, 'Debugger.enable', {});
    // https://chromedevtools.github.io/devtools-protocol/tot/Tracing/#method-start
    await chrome.debugger.sendCommand(
      { tabId }, // target
      'Tracing.start', // method 
      { transferMode: 'ReportEvents', traceConfig: TRACE_CONFIG }); // command params
  } catch (error) {
    console.error('Error in startTracing:', error);
  }
}

  /**
   * Stop the tracing session.
   * @param {number} tabId the Tab for which to stop the tracing.
   */
  async function stopTracing(tabId) {
    // https://chromedevtools.github.io/devtools-protocol/tot/Debugger/#method-disable
    await chrome.debugger.sendCommand({ tabId }, 'Debugger.disable', {});
    // https://chromedevtools.github.io/devtools-protocol/tot/Tracing/#method-end
    await chrome.debugger.sendCommand({ tabId }, 'Tracing.end', {});
  }

  /**
   * Process a high entropy api trace and get the source code of the script.
   * @param {!object} trace The trace.
   * @param {number} tabId the Tab ID.
   */
  async function processTrace(trace, tabId) {
    // https://source.chromium.org/chromium/chromium/src/+/main:base/tracing/protos/chrome_track_event.proto;l=1099;drc=ab69f2fe4dfdfc4268953b5d42c7019e28f0b0cd
    if (!(trace.cat === HIGH_ENTROPY_API_CATEGORY && trace.args &&
      trace.args.high_entropy_api && trace.args.high_entropy_api.called_api))
      return;

    const context = trace.args.high_entropy_api.execution_context;
    const apiIdentifier = trace.args.high_entropy_api.called_api.identifier;
    const sourceLocation = trace.args.high_entropy_api.source_location;
    const top_level_url = context.url;
    const args = trace.args.high_entropy_api.called_api.func_arguments;
    const scriptId = sourceLocation.script_id;
    const sourceMapURL = sourceLocation.sourceMapURL;

    // Check if script is from an eval call
    const isEvalScript = !sourceLocation.url || sourceLocation.url.startsWith('eval://');
    let scriptURLToUse = isEvalScript && sourceMapURL ? sourceMapURL : sourceLocation.url;
    if (!(apiIdentifier && context && sourceLocation)) return;
  
    // Exclude High Entropy APIs called by Devtools.
    // https://crsrc.org/c/third_party/devtools-frontend/src/inspector_overlay/
    if (sourceLocation && context.origin === 'null' && context.url === '' && [
      'setCanvas', 'resetCanvas', 'drawViewSize'
    ].includes(sourceLocation.function_name))
      return;

    // Filter out traces caused by extensions.
    if (sourceLocation && scriptURLToUse &&
      scriptURLToUse.startsWith('chrome-extension://'))
      return;
    
    
    // call the server to record data
    fetch(`http://localhost:${PORT}/apiTraces`, {
      method: "POST",
      body: JSON.stringify({
        "top_level_url": top_level_url,
        "scriptId": scriptId,
        "scriptURL": scriptURLToUse,
        "API": apiIdentifier,
        "Args": args,
        "functionName": sourceLocation.function_name,
        "lineNumber:columnNumber": sourceLocation.line_number + ":" + sourceLocation.column_number
      }),
      mode: 'cors',
      headers: {
        'Access-Control-Allow-Origin': '*',
        "Content-Type": "application/json"
      }
    }).then(res => {
      console.log("Response complete! response");
    });
  }
  
  async function handleScriptParsed(tabId, params) {
    try {
      const { scriptId, url, sourceMapURL, executionContextId } = params;
      let scriptUrlToUse = url && url.trim() !== '' ? url : sourceMapURL;

      // get script id
      const { scriptSource } = await chrome.debugger.sendCommand(
        { tabId }, 
        'Debugger.getScriptSource', 
        { scriptId }
      );
  
      if (scriptSource) {
        let scriptIdInt = parseInt(scriptId)
        await fetch(`http://localhost:${PORT2}/scriptSources`, {
          method: "POST",
          body: JSON.stringify({
            "scriptId": scriptIdInt, 
            "scriptURL": scriptUrlToUse, 
            "scriptSource": scriptSource
          }),
          mode: 'cors',
          headers: {
            'Access-Control-Allow-Origin': '*',
            "Content-Type": "application/json"
          }
        });
        console.log("Capture the script source");
      } else {
        console.log("Script source not found for Script Id:", scriptId);
      }
    } catch (error) {
      console.error('Error fetching script source:', error);
    }
  }
  
  // Fired whenever debugging target issues instrumentation event
  chrome.debugger.onEvent.addListener(async (source, method, params) => {
    switch (method) {
      case 'Debugger.scriptParsed': // Fired when virtual machine parses script
        // https://chromedevtools.github.io/devtools-protocol/tot/Debugger/#event-scriptParsed
        if (!(params.url.startsWith('chrome-extension://'))) {
          // console.log(`ScriptId ${params.scriptId} parsed (${params.url})`);
          handleScriptParsed(source.tabId, params);
        }
        break;
      case 'Tracing.dataCollected':
        // https://chromedevtools.github.io/devtools-protocol/tot/Tracing/#event-dataCollected
        params.value.forEach(trace => processTrace(trace, source.tabId));
        // params.value.forEach(processTrace);
        break;
      case 'Tracing.tracingComplete':
        // https://chromedevtools.github.io/devtools-protocol/tot/Tracing/#event-tracingComplete
        console.log('Tracing complete and debugger detached.');
        await chrome.debugger.detach({ tabId: source.tabId });
        break;
    }
  });

  // Triggered before navigation starts
  chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
    if (!details.tabId || details.tabId < 0) return;

    await startTracing(details.tabId);
    console.log('Tracing started.');
    console.log(`Tracing for ${TRACING_TIME_MILLISECONDS} milliseconds...`);
    setTimeout(() => stopTracing(details.tabId), TRACING_TIME_MILLISECONDS);
  });