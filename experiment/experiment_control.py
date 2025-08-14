from env import (
    GOOGLE_PATH,
    GITHUB_AUTH_TKN,
    SLACK_CONFIG_OBJECT,
    NOTION_CONFIG_OBJECT,
    PERPLEXITY_CONFIG_OBJECT,
    BROWSERBASE_CONFIG_OBJECT,
    LINEAR_CONFIG_OBJECT,
    STDIO_PATH,
    ACCUWEATHER_CONFIG_OBJECT,
    BRAVE_CONFIG_OBJECT,
)

retry_count = 5

# Browsing & Search MCPs
BROWSING_SEARCH_MCPS = {
    "browserbase": {
        "command": "npx",
        "args": ["@browserbasehq/mcp"],
        "env": BROWSERBASE_CONFIG_OBJECT,
    },
    "perplexity-search": {
        "env": PERPLEXITY_CONFIG_OBJECT,
        "command": "uvx",
        "args": ["perplexity-mcp"],
    },
    "wikipedia": {"command": "uvx", "args": ["wikipedia-mcp"]},
    "weather": {
        "command": "npx",
        "args": ["-y", "@timlukahorstmann/mcp-weather"],
        "env": ACCUWEATHER_CONFIG_OBJECT,
    },
    "arxiv-mcp-server": {
        "command": "uv",
        "args": [
            "tool",
            "run",
            "arxiv-mcp-server",
            "--storage-path", STDIO_PATH + "/paper_storage"
        ]
    },
    "brave-search": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-brave-search"
      ],
      "env": BRAVE_CONFIG_OBJECT
    },
}

# Productivity & Info Gathering MCPs
PRODUCTIVITY_INFO_MCPS = {
    "github": {
        "command": "docker",
        "args": [
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server",
        ],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_AUTH_TKN},
    },
    "linear": {
        "command": "npx",
        "args": ["-y", "@tacticlaunch/mcp-linear"],
        "env": LINEAR_CONFIG_OBJECT,
    },
    "google-toolbox": {
        "command": "uv",
        "args": ["--directory", GOOGLE_PATH, "run", "server.py"],
    },
    "slack": {
        "command": "npx",
        "args": ["-y", "slack-mcp-server@latest", "--transport", "stdio"],
        "env": SLACK_CONFIG_OBJECT,
    },
    "notionApi": {
        "command": "npx",
        "args": ["-y", "@notionhq/notion-mcp-server"],
        "env": NOTION_CONFIG_OBJECT,
    },
}

# Programming & Code MCPs
PROGRAMMING_CODE_MCPS = {
    "context7": {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]},
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        STDIO_PATH,
        STDIO_PATH + "/test_files"
      ]
    },
    "mcp-web-fetch": {"command": "uvx", "args": ["mcp-web-fetch"]},
    "mcp-python-code-execution": {
        "command": "uvx",
        "args": [
            "--from",
            "git+https://github.com/pathintegral-institute/mcp.science@main#subdirectory=servers/python-code-execution",
            "mcp-python-code-execution",
        ],
        "env": {},
    },
}

# Visual & Chart MCPs
VISUAL_CHART_MCPS = {
    "mcp-server-chart": {
        "command": "npx",
        "args": ["-y", "@antv/mcp-server-chart"],
    },
    "visual-reasoning": {
        "command": "npx",
        "args": ["-y", "@waldzellai/visual-reasoning"],
    },
    "shadcn-ui-server": {
        "command": "npx",
        "args": ["-y", "shadcn-ui-mcp-server"],
    },
}

# Extra MCPs
EXTRA_MCPS = {
    "calculator": {"command": "uvx", "args": ["mcp-server-calculator"]},
}

# All MCP servers combined
ALL_MCP_SERVERS = {
    **BROWSING_SEARCH_MCPS,
    **PRODUCTIVITY_INFO_MCPS,
    **PROGRAMMING_CODE_MCPS,
    **VISUAL_CHART_MCPS,
    **EXTRA_MCPS,
}

CANDIDATE_MODELS = [
    "moonshotai/kimi-k2",
    "anthropic/claude-sonnet-4",
    "openai/o3",
    "google/gemini-2.5-pro",
    "openai/gpt-oss-120b"
    "openai/gpt-5"
]

test_batteries = [
    {
        "name": "test_pretrain_knowledge_with_no_mcp_servers",
        # testing if models can properly select the right tool when all other mcps are connected
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": {},
        "queries": {
            "What animal represents the year 2000 on the Chinese calendar?": "dragon",
            "Which of the following is part of a farm combine? (Beater) (Plough) (Stalker) (Macerator)": "Beater",
            "What is a Dutch oven?": "a kettle pot",
            "A completely submerged object always displaces its own": "volume of fluid.",
            "A voltage will be induced in a wire loop when the magnetic field within that loop": "changes",
            "According to general relativity,": "All of these.",
            "A step-up transformer in an electric circuit can step up": "voltage",
            "A wave transfers": "energy",
            "Immediately after two separated charged particles are released from rest, both increase in acceleration. The sign of charge of the particles is therefore": "opposite",
            "The surface of planet Earth loses energy to outer space due mostly to": "radiation",
            "In a series circuit, if the current in one lamp is 2 A, the current in the lamp next to it is": "2 A",
            "As a blinking light source approaching you gains speed, you see the frequency of flashes": "increase.",
            "In the absence of air resistance a ball of mass m is tossed upward to reach a height of 20 m. At the 10-m position half way up the net force on the ball is": "mg",
            "Pumice is a volcanic rock that floats in water. The density of pumice compared with that of water is": "less.",
            "When you look at the red petals of a rose the color light you're seeing is": "red.",
            "Compared with the sound you hear from the siren of a stationary fire engine the sound you hear when it approaches you has an increased": "frequency",
            "When you squeeze an air-filled party balloon, you increase its": "density",
            "Polarization is a property of": "transverse waves",
            "In a vacuum, an object has no": "buoyant force",
            "Temperature is generally proportional to a substance’s": "average translational kinetic energy.",
            "In a hydraulic press operation, the output piston cannot": "produce increased energy",
            "When a spinning system contracts in the absence of an external torque its rotational speed increases and its angular momentum": "remains unchanged",
            "According to Hooke’s law, if you hang by a tree branch and note how much it bends, then hanging with twice the weight produces": "twice the bend.",
            "When an increase in speed doubles the kinetic energy of a moving body its momentum": "increases but less than doubles",
            "If an object is already moving and the sum of all the vector forces on a mass is zero, then the object will": "move at a constant speed in a straight line",
            "Relativity equations for time, length, and momentum hold true for": "Both of these.",
            "A heavy rock and a light rock in free fall (zero air resistance) have the same acceleration. The heavy rock doesn't have a greater acceleration because the": "ratio of force to mass is the same.",
            "London dispersion forces are caused by": "temporary dipoles created by the position of electrons around the nuclei in a molecule",
            "An unknown substance is found to have a high melting point. In addition, it is a poor conductor of electricity and does not dissolve in water. The substance most likely contains": "covalent network bonding",
            "Hund's rule requires that": "no two electrons can pair up if there is an empty orbital at the same energy level available",
            "Sulfurous acid is a weak acid, while sulfuric acid is a much stronger acid because": "the O–H bonds in sulfuric acid are much weaker than in sulfurous acid due to the electron withdrawing of the oxygen atoms on sulfuric acid",
            "Which of the following substances has an asymmetrical molecular structure?": "SF4",
            "Hydrogen fluoride, HF, is a liquid at 15°C. All other hydrogen halides (represented by HX, where X is any other halogen) are gases at the same temperature. Why?": "The dipoles in a HF molecule exhibit a particularly strong attraction force to the dipoles in other HF molecules.",
            "A sample of liquid NH3 is brought to its boiling point. Which of the following occurs during the boiling process?": "The hydrogen bonds holding separate NH3 molecules together break apart.",
            "A student has a liter of a 0.100 M solution of a strong acid. To prepare a buffer, this should be mixed with": "a weak base",
            "Which of the following indicates that a reaction is spontaneous?": "at equilibrium there are more products than reactants",
            "A mechanism is a sequence of elementary reactions that add up to the overall reaction stoichiometry. A substance that is produced in one elementary reaction and consumed in another is called": "an intermediate",
            "The ideal gas law is successful for most gases because": "gas particles do not interact significantly",
            "The bond length between any two nonmetal atoms is achieved under which of the following conditions?": "Where the energy of interaction between the atoms is at its minimum value",
            "The melting point of straight-chain hydrocarbons increases as the number of carbon atoms increase. The reason for this is the": "increasing number of induced dipoles per molecule",
            "Which would be the easiest way to burn an iron nail?": "Grind the nail into very small, dust-sized particles, and spray them into a flame",
            "NH4+(aq) + NO2- (aq) → N2(g) + 2H2O(l) Increasing the temperature of the above reaction will increase the rate of reaction. Which of the following is NOT a reason that increased temperature increases reaction rate?": "Alternate reaction pathways become available at higher temperatures.",
            "The relationship between the vapor pressure of a liquid and the heat of vaporization is expressed in the": "Clausius-Clapeyron equation",
            "Which following pair of substances can be used to make a buffer solution?": "HC2H3O2 and KC2H3O2",
            "When will Kp and Kc have the same numerical value?": "When the reaction exhibits no change in pressure at constant volume",
            "PCl3(g) + Cl2(g) ↔ PCl5(g) ΔH = -92.5 kJ/mol In which of the following ways could the reaction above be manipulated to create more product?": "Increasing the pressure",
            "A 25 g sample of a liquid was heated to 100 °C and then quickly transferred to an insulated container holding 100 g of water at 22 °C. The temperature of the mixture rose to reach a final temperature of 35 °C. Which of the following can be concluded?": "The sample temperature changed more than the water temperature did, but the sample lost the same amount of heat energy as the water gained.",
            "A sample of oxygen gas at 50 °C is heated, reaching a final temperature of 100 °C. Which statement best describes the behavior of the gas molecules?": "Their kinetic energy increases by a factor of less than 2.",
            "Under which conditions will a real gas most closely behave as an ideal gas?": "High temperature and low pressure",
            "When an ideal gas is allowed to expand isothermally, which one of the following is true?": "q = -w",
            "In chemistry an ideal solution is a": "mixture where the potential energy of the mixture is the same as that of the individual solvents",
            "For Socrates, an unexamined life is a tragedy because it results in grievous harm to _____.": "the soul",
            "According to Kant, nothing can be called “good” without qualification except _____.": "a good will",
            "In Aristotle’s terminology, incontinence is when:": "one knows that one’s actions are wrong, but does them anyway.",
            "Nagel claims that most skeptical arguments:": "grow from the consistent application of ordinary standards.",
            "Rawls conceives of the original contract as one to:": "establish the principles of justice for the basic structure of society.",
            "According to Singer, compliance with his principle requires:": "drastic changes to both our personal habits and to our society.",
            "For Socrates, the soul is harmed by lack of _____.": "knowledge",
            "According to Kant, morality requires us to:": "act only on maxims that we can will to become universal laws.",
            "According to Gauthier, the basis of morality is:": "the agreement of rational persons choosing the terms of their interaction.",
            "Anscombe claims that it is not profitable to do moral philosophy until we have an adequate philosophy of:": "psychology.",
            "Mill claims that one of the strongest objections to utilitarianism is drawn from the idea of:": "justice.",
            "Berkeley asserts that existing and perceiving are _____.": "one and the same thing",
            "According to Rawls, the term “justice as fairness” conveys the idea that the principles of justice are agreed to in an initial position that is _____.": "fair",
            "Berkeley believes that sensible things cannot exist except in _____.": "a mind",
            "Singer claims that his argument upsets the traditional distinction between:": "duty and charity.",
            "In his discussion of the Divided Line, Plato says that, in contrast to mere belief or opinion, knowledge is a belief for which we give reasons or justifications by appealing:": "beyond sense experience to unchanging ideas (Forms) that are perceived as rationally ordered.",
            "Stevenson’s primary aim in this paper is to:": "make ethical questions clear.",
            "According to Moore, we are thinking about good whenever we think about:": "intrinsic value.",
            "Baier argues that genuine moral rules:": "must be for the good of human beings.",
            "Nussbaum claims that to many current ethical theorists, turning to an ethical approach based on the virtues is connected with a turn toward:": "relativism.",
            "Philosophy is concerned primarily with identifying beliefs about human existence and evaluating arguments that support those beliefs. These activities can be summarized in two questions that drive philosophical investigations:": "what do you mean? and how do you know?",
            "Epictetus claims that things within our power are __________ and things not in our power are __________.": "free and unhindered; servile and subject to hindrance",
            "Hume divides our perceptions into two categories:": "impressions and ideas.",
            "Aristotle says that what makes things be what they are--their essence--does not exist apart from individ-uals that exist in the world. So if all the members of a species were destroyed, then their essence or form:": "would likewise be destroyed.",
        },
    },
    {
        "name": "test_pretrain_knowledge_with_all_mcp_servers",
        # testing if models can properly select the right tool when all other mcps are connected
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": ALL_MCP_SERVERS,
        "queries": {
            "What animal represents the year 2000 on the Chinese calendar?": "dragon",
            "Which of the following is part of a farm combine? (Beater) (Plough) (Stalker) (Macerator)": "Beater",
            "What is a Dutch oven?": "a kettle pot",
            "A completely submerged object always displaces its own": "volume of fluid.",
            "A voltage will be induced in a wire loop when the magnetic field within that loop": "changes",
            "According to general relativity,": "All of these.",
            "A step-up transformer in an electric circuit can step up": "voltage",
            "A wave transfers": "energy",
            "Immediately after two separated charged particles are released from rest, both increase in acceleration. The sign of charge of the particles is therefore": "opposite",
            "The surface of planet Earth loses energy to outer space due mostly to": "radiation",
            "In a series circuit, if the current in one lamp is 2 A, the current in the lamp next to it is": "2 A",
            "As a blinking light source approaching you gains speed, you see the frequency of flashes": "increase.",
            "In the absence of air resistance a ball of mass m is tossed upward to reach a height of 20 m. At the 10-m position half way up the net force on the ball is": "mg",
            "Pumice is a volcanic rock that floats in water. The density of pumice compared with that of water is": "less.",
            "When you look at the red petals of a rose the color light you're seeing is": "red.",
            "Compared with the sound you hear from the siren of a stationary fire engine the sound you hear when it approaches you has an increased": "frequency",
            "When you squeeze an air-filled party balloon, you increase its": "density",
            "Polarization is a property of": "transverse waves",
            "In a vacuum, an object has no": "buoyant force",
            "Temperature is generally proportional to a substance’s": "average translational kinetic energy.",
            "In a hydraulic press operation, the output piston cannot": "produce increased energy",
            "When a spinning system contracts in the absence of an external torque its rotational speed increases and its angular momentum": "remains unchanged",
            "According to Hooke’s law, if you hang by a tree branch and note how much it bends, then hanging with twice the weight produces": "twice the bend.",
            "When an increase in speed doubles the kinetic energy of a moving body its momentum": "increases but less than doubles",
            "If an object is already moving and the sum of all the vector forces on a mass is zero, then the object will": "move at a constant speed in a straight line",
            "Relativity equations for time, length, and momentum hold true for": "Both of these.",
            "A heavy rock and a light rock in free fall (zero air resistance) have the same acceleration. The heavy rock doesn't have a greater acceleration because the": "ratio of force to mass is the same.",
            "London dispersion forces are caused by": "temporary dipoles created by the position of electrons around the nuclei in a molecule",
            "An unknown substance is found to have a high melting point. In addition, it is a poor conductor of electricity and does not dissolve in water. The substance most likely contains": "covalent network bonding",
            "Hund's rule requires that": "no two electrons can pair up if there is an empty orbital at the same energy level available",
            "Sulfurous acid is a weak acid, while sulfuric acid is a much stronger acid because": "the O–H bonds in sulfuric acid are much weaker than in sulfurous acid due to the electron withdrawing of the oxygen atoms on sulfuric acid",
            "Which of the following substances has an asymmetrical molecular structure?": "SF4",
            "Hydrogen fluoride, HF, is a liquid at 15°C. All other hydrogen halides (represented by HX, where X is any other halogen) are gases at the same temperature. Why?": "The dipoles in a HF molecule exhibit a particularly strong attraction force to the dipoles in other HF molecules.",
            "A sample of liquid NH3 is brought to its boiling point. Which of the following occurs during the boiling process?": "The hydrogen bonds holding separate NH3 molecules together break apart.",
            "A student has a liter of a 0.100 M solution of a strong acid. To prepare a buffer, this should be mixed with": "a weak base",
            "Which of the following indicates that a reaction is spontaneous?": "at equilibrium there are more products than reactants",
            "A mechanism is a sequence of elementary reactions that add up to the overall reaction stoichiometry. A substance that is produced in one elementary reaction and consumed in another is called": "an intermediate",
            "The ideal gas law is successful for most gases because": "gas particles do not interact significantly",
            "The bond length between any two nonmetal atoms is achieved under which of the following conditions?": "Where the energy of interaction between the atoms is at its minimum value",
            "The melting point of straight-chain hydrocarbons increases as the number of carbon atoms increase. The reason for this is the": "increasing number of induced dipoles per molecule",
            "Which would be the easiest way to burn an iron nail?": "Grind the nail into very small, dust-sized particles, and spray them into a flame",
            "NH4+(aq) + NO2- (aq) → N2(g) + 2H2O(l) Increasing the temperature of the above reaction will increase the rate of reaction. Which of the following is NOT a reason that increased temperature increases reaction rate?": "Alternate reaction pathways become available at higher temperatures.",
            "The relationship between the vapor pressure of a liquid and the heat of vaporization is expressed in the": "Clausius-Clapeyron equation",
            "Which following pair of substances can be used to make a buffer solution?": "HC2H3O2 and KC2H3O2",
            "When will Kp and Kc have the same numerical value?": "When the reaction exhibits no change in pressure at constant volume",
            "PCl3(g) + Cl2(g) ↔ PCl5(g) ΔH = -92.5 kJ/mol In which of the following ways could the reaction above be manipulated to create more product?": "Increasing the pressure",
            "A 25 g sample of a liquid was heated to 100 °C and then quickly transferred to an insulated container holding 100 g of water at 22 °C. The temperature of the mixture rose to reach a final temperature of 35 °C. Which of the following can be concluded?": "The sample temperature changed more than the water temperature did, but the sample lost the same amount of heat energy as the water gained.",
            "A sample of oxygen gas at 50 °C is heated, reaching a final temperature of 100 °C. Which statement best describes the behavior of the gas molecules?": "Their kinetic energy increases by a factor of less than 2.",
            "Under which conditions will a real gas most closely behave as an ideal gas?": "High temperature and low pressure",
            "When an ideal gas is allowed to expand isothermally, which one of the following is true?": "q = -w",
            "An ideal solution is a": "mixture where the potential energy of the mixture is the same as that of the individual solvents",
            "For Socrates, an unexamined life is a tragedy because it results in grievous harm to _____.": "the soul",
            "According to Kant, nothing can be called “good” without qualification except _____.": "a good will",
            "In Aristotle’s terminology, incontinence is when:": "one knows that one’s actions are wrong, but does them anyway.",
            "Nagel claims that most skeptical arguments:": "grow from the consistent application of ordinary standards.",
            "Rawls conceives of the original contract as one to:": "establish the principles of justice for the basic structure of society.",
            "According to Singer, compliance with his principle requires:": "drastic changes to both our personal habits and to our society.",
            "For Socrates, the soul is harmed by lack of _____.": "knowledge",
            "According to Kant, morality requires us to:": "act only on maxims that we can will to become universal laws.",
            "According to Gauthier, the basis of morality is:": "the agreement of rational persons choosing the terms of their interaction.",
            "Anscombe claims that it is not profitable to do moral philosophy until we have an adequate philosophy of:": "psychology.",
            "Mill claims that one of the strongest objections to utilitarianism is drawn from the idea of:": "justice.",
            "Berkeley asserts that existing and perceiving are _____.": "one and the same thing",
            "According to Rawls, the term “justice as fairness” conveys the idea that the principles of justice are agreed to in an initial position that is _____.": "fair",
            "Berkeley believes that sensible things cannot exist except in _____.": "a mind",
            "Singer claims that his argument upsets the traditional distinction between:": "duty and charity.",
            "In his discussion of the Divided Line, Plato says that, in contrast to mere belief or opinion, knowledge is a belief for which we give reasons or justifications by appealing:": "beyond sense experience to unchanging ideas (Forms) that are perceived as rationally ordered.",
            "Stevenson’s primary aim in this paper is to:": "make ethical questions clear.",
            "According to Moore, we are thinking about good whenever we think about:": "intrinsic value.",
            "Baier argues that genuine moral rules:": "must be for the good of human beings.",
            "Nussbaum claims that to many current ethical theorists, turning to an ethical approach based on the virtues is connected with a turn toward:": "relativism.",
            "Philosophy is concerned primarily with identifying beliefs about human existence and evaluating arguments that support those beliefs. These activities can be summarized in two questions that drive philosophical investigations:": "what do you mean? and how do you know?",
            "Epictetus claims that things within our power are __________ and things not in our power are __________.": "free and unhindered; servile and subject to hindrance",
            "Hume divides our perceptions into two categories:": "impressions and ideas.",
            "Aristotle says that what makes things be what they are--their essence--does not exist apart from individ-uals that exist in the world. So if all the members of a species were destroyed, then their essence or form:": "would likewise be destroyed.",
        },
    },

    {
        "name": "test_tool_usage_browsing_with_all_mcp_servers",
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": ALL_MCP_SERVERS,
        "queries": {
            # Basic tool access tests
            "What is the current weather in New York City?": "The response should use AccuWeather MCP to get current weather conditions for NYC.",
            "Who is the best basketball player of all time?": "The response should use some mcp tool to find the best basketball player of all time.",
            "How expensive is a bag of Doritos right now?": "The response should use some mcp tool to find the price of a bag of Doritos.",
            "What is the national debt as it stands right now?": "The response should use some mcp tool to find the national debt.",
            "Where is the cheapest flight from new york to anywhere in the world?": "The response should use some mcp tool to find the cheapest flight from new york to anywhere in the world.",
            # Complex synthesis tests (existing)
            "find me the best rated wireless headphones under $200 on amazon": "The response should use Amazon Product Search MCP to find highly-rated wireless headphones under $200, providing specific product recommendations with ratings and prices.",
            "Search for the book 'A Princess of Mars' by Edgar Rice Burroughs on Wikipedia. What is the name of the protagonist's companion?": "The response should use web browsing to find the Wikipedia page for 'A Princess of Mars' and extract the name of the protagonist's companion.",
            "I need to plan a camping trip to Yellowstone National Park next month. Find me: 1) The current weather forecast and typical weather patterns, 2) Research recent scientific papers about wildlife behavior in Yellowstone, 3) Find camping gear recommendations on Amazon (tent, sleeping bag, hiking boots) under $500 total, and 4) Look up park regulations and best camping spots on Wikipedia. Synthesize this into a comprehensive camping plan.": "The response should use multiple MCP servers: AccuWeather for weather data, arXivPaper for scientific research on Yellowstone wildlife, Amazon for camping gear recommendations with budget constraints, Wikipedia for park information, and a search engine for additional park regulations. The model should synthesize all information into a coherent camping plan.",
            "Compare the academic consensus on climate change impacts on coral reefs with current consumer attitudes toward eco-friendly products. Find 1) Recent peer-reviewed papers on coral reef bleaching and climate change from arXiv, 2) Use a search engine to search for recent surveys on consumer eco-consciousness, 3) Find the best-selling eco-friendly products on Amazon, and 4) Research the Great Barrier Reef's current status on Wikipedia. Create a comprehensive analysis connecting scientific findings to market trends.": "The response should demonstrate complex information synthesis by using arXivPaper for scientific research, a search engine for consumer behavior data, Amazon for market trends, and Wikipedia for current environmental status. The model should create connections between scientific evidence and consumer behavior patterns.",
            "Investigate the relationship between renewable energy adoption and economic growth. 1) Search for recent academic papers on renewable energy economics, 2) Use a search engine to find current global renewable energy statistics and government policies, 3) Look up major renewable energy companies and their stock performance on financial sites through web browsing, 4) Find renewable energy investment products on Amazon (solar panels, energy monitoring devices), and 5) Research specific country case studies on Wikipedia (Denmark, Costa Rica, etc.). Provide a comprehensive analysis with policy recommendations.": "The response should demonstrate advanced analytical thinking by combining academic research (arXivPaper), current data (a search engine), market analysis (web browsing), consumer products (Amazon), and case studies (Wikipedia). The model should synthesize complex economic and environmental data into actionable insights.",
        },
    },
    {
        "name": "test_tool_usage_info_gathering_with_all_mcp_servers",
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": ALL_MCP_SERVERS,
        "queries": {
            # Basic tool access tests
            "List my GitHub repositories": "The response should use GitHub MCP to list the user's repositories with basic information like name and description.",
            "Show me the first 5 Linear issues": "The response should use Linear MCP to retrieve and display the first 5 issues with their titles and status.",
            "List the first 5 emails in my Gmail inbox": "The response should use Google Toolbox MCP to access Gmail and show the first 5 email subjects and senders.",
            "Show me recent messages from my Slack workspace": "The response should use Slack MCP to retrieve and display recent messages from available channels.",
            "List my Notion pages": "The response should use Notion MCP to show available pages and databases.",
            # Complex integration tests (existing)
            "can you list the files in my google drive? and if there exists a file identical to any in tanderson1299320's repository jSpreed let me know": "The response should use Google Drive MCP to list files and GitHub MCP to access the jSpreed repository, then compare files to identify any matches between the two locations.",
            "Based on the content of the files in the 'test_files' directory, can you identify which file is most likely to be a dataset of chemical elements and then find the discoverer of the first metal listed with a null molar heat?": "The response should use file system tools to list and read files in the 'test_files' directory, identify the JSON file with chemical elements, parse it, and find the requested information.",
            "There is a discussion in slack about adding features to the golang speedreader. Can you find that discussion and summarize the features that are being requested?": "The response should use Slack MCP to search for discussions about the 'golang speedreader' and summarize the requested features.",
            "There is an issue in our github repo about enriching the readme of the golang speedreader. Can you find it and tell me who is assigned to it?": "The response should use GitHub MCP to find the specified issue in the repository and identify the assignee.",
            "Can you compare all the files in my local test_files directory with the files in my Google Drive root folder and tell me which files exist in both locations, which are only local, and which are only in Google Drive? For any files that exist in both locations, check if their sizes or modification dates differ.": "The response should use file system tools to list and analyze files in the local 'test_files' directory, use Google Drive MCP to list files in the Drive root folder, then perform a comprehensive comparison of file names, sizes, and timestamps between the two locations.",
        },
    },
    {
        "name":"test_tool_usage_programming_with_all_mcp_servers",
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": ALL_MCP_SERVERS,
        "queries": {
            # Basic tool access tests
            "List the files in the test_files directory": "The response should use filesystem MCP to list the contents of the test_files directory.",
            "Execute a simple Python script that prints 'Hello World'": "The response should use Python code execution MCP to run a basic print statement.",
            "Fetch the content from https://httpbin.org/json": "The response should use web fetch MCP to retrieve JSON data from the test endpoint.",
            "Access the jSpreed repository on GitHub and show me the README": "The response should use GitHub MCP to access the specified repository and display README content.",
            "Search Context7 for documentation about 'React hooks'": "The response should use Context7 MCP to search for React hooks documentation.",
            # Complex programming tasks (existing)
            "implement the changes to our go speedreader program at https://raw.githubusercontent.com/tanderson1299320/jSpreed/refs/heads/main/main.go specified in our linear": "The response should use GitHub MCP to access the Go code, Slack MCP or Linear-related tools to find specifications, and provide specific code implementation changes.",
            "re-write the go speedreader program at https://raw.githubusercontent.com/tanderson1299320/jSpreed/refs/heads/main/main.go in python and while doing so address the concerns about it in the slack": "The response should use GitHub MCP to get the Go code, Slack MCP to find concerns, and provide a complete Python rewrite that addresses the identified issues.",
            "list for me the newest possible versions of the packages in https://raw.githubusercontent.com/tanderson1299320/Gulp-AI/Osmosis-MCP-4B-demo/refs/heads/main/requirements.txt": "The response should use GitHub MCP to access the requirements.txt file and search tools to find the latest available versions of each listed package.",
            "In the golang speedreader program at https://raw.githubusercontent.com/tanderson1299320/jSpreed/refs/heads/main/main.go, add a function for calculating WPM based on a counter that starts when the program does": "The response should use GitHub MCP to get the Go code and provide a modified Go function that calculates both WPM and accuracy.",
            "The file 'test_files/d6059b3e-e1da-43b4-ac26-ecad2984909b.csv' contains data about LEGO parts. Can you write a python script that reads this file and creates a new CSV file containing only the parts that are black?": "The response should provide a Python script that uses a library like pandas or csv to read the specified CSV file, filter for rows where the color is 'Black', and write the result to a new CSV file.",
        },
    },
    {
        "name": "test_tool_usage_visual_with_all_mcp_servers",
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": ALL_MCP_SERVERS,
        "queries": {
            # Basic tool access tests
            "Create a simple bar chart with sample data (apples: 10, oranges: 15, bananas: 8) with a chart mcp": "The response should contain a link to the chart",
            "Create a simple bar chart with sample data (apples: 10, oranges: 15, bananas: 8)": "The response should contain a link to the chart",
            "List the files in my Google Drive": "The response should use Google Toolbox MCP to access Google Drive and list available files.",
            "Navigate to https://crouton.net using the browser": "The response should use Browserbase MCP to navigate to the example website.",
            "Generate a simple UI component with ShadCN": "The response should use ShadCN UI MCP to create a basic UI component.",
            # Complex visual tasks (existing)
            "List files in my google drive then create a bar chart showing the number of types of files. Then look at the resulting image and tell me the trendline": "The response should use Google Drive MCP to list files, Chart MCP to create a bar chart of file types, and Visual Reasoning MCP to analyze the chart and describe file type distribution trends.",
            "Graph the dates of the emails sent to my gmail using the google-toolbox. Then look at the resulting image and tell me the trendline": "The response should contain trend information about the emails in my inbox.",
            "make a chart of computer color capabilites by year based on the data in https://en.wikipedia.org/wiki/List_of_color_palettes": "The response should use Browserbase MCP to access the Wikipedia page, extract color capability data by year, there should be a link to the chart in the response",
            "chart the distance relation between angkor wat, the roman colosseum, and tenochtitlan": "The response should use Google Maps MCP to get location coordinates, Chart MCP to create a distance visualization (possibly a triangle chart or distance matrix), and Visual Reasoning MCP to analyze the geographic relationships.",
            "Please take the data from 'test_files/7805912b-c8da-4134-9b54-b590f884352a.csv' and create a pie chart showing the precipitation distribution in the different boroughs of New York. Then, tell me which borough has the smallest share of precipitation.": "The response should use file system tools to read the CSV, Chart MCP to create a pie chart, and Visual Reasoning MCP to analyze the chart and identify the borough with the smallest precipitation share.",
        },
    },

    {
        "name": "test_tool_usage_with_only_browsing_mcp_servers",
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": BROWSING_SEARCH_MCPS,
        "queries": {
            # Basic tool access tests
            "What is the current weather in New York City?": "The response should use AccuWeather MCP to get current weather conditions for NYC.",
            "Who is the best basketball player of all time?": "The response should use some mcp tool to find the best basketball player of all time.",
            "How expensive is a bag of Doritos right now?": "The response should use some mcp tool to find the price of a bag of Doritos.",
            "What is the national debt as it stands right now?": "The response should use some mcp tool to find the national debt.",
            "Where is the cheapest flight from new york to anywhere in the world?": "The response should use some mcp tool to find the cheapest flight from new york to anywhere in the world.",
            # Complex synthesis tests (existing)
            "find me the best rated wireless headphones under $200 on amazon": "The response should use Amazon Product Search MCP to find highly-rated wireless headphones under $200, providing specific product recommendations with ratings and prices.",
            "Search for the book 'A Princess of Mars' by Edgar Rice Burroughs on Wikipedia. What is the name of the protagonist's companion?": "The response should use web browsing to find the Wikipedia page for 'A Princess of Mars' and extract the name of the protagonist's companion.",
            "I need to plan a camping trip to Yellowstone National Park next month. Find me: 1) The current weather forecast and typical weather patterns, 2) Research recent scientific papers about wildlife behavior in Yellowstone, 3) Find camping gear recommendations on Amazon (tent, sleeping bag, hiking boots) under $500 total, and 4) Look up park regulations and best camping spots on Wikipedia. Synthesize this into a comprehensive camping plan.": "The response should use multiple MCP servers: AccuWeather for weather data, arXivPaper for scientific research on Yellowstone wildlife, Amazon for camping gear recommendations with budget constraints, Wikipedia for park information, and potentially a search engine for additional park regulations. The model should synthesize all information into a coherent camping plan.",
            "Compare the academic consensus on climate change impacts on coral reefs with current consumer attitudes toward eco-friendly products. Find 1) Recent peer-reviewed papers on coral reef bleaching and climate change from arXiv, 2) Use a search engine to search for recent surveys on consumer eco-consciousness, 3) Find the best-selling eco-friendly products on Amazon, and 4) Research the Great Barrier Reef's current status on Wikipedia. Create a comprehensive analysis connecting scientific findings to market trends.": "The response should demonstrate complex information synthesis by using arXivPaper for scientific research, a search engine for consumer behavior data, Amazon for market trends, and Wikipedia for current environmental status. The model should create connections between scientific evidence and consumer behavior patterns.",
            "Investigate the relationship between renewable energy adoption and economic growth. 1) Search for recent academic papers on renewable energy economics, 2) Use a search engine to find current global renewable energy statistics and government policies, 3) Look up major renewable energy companies and their stock performance on financial sites through web browsing, 4) Find renewable energy investment products on Amazon (solar panels, energy monitoring devices), and 5) Research specific country case studies on Wikipedia (Denmark, Costa Rica, etc.). Provide a comprehensive analysis with policy recommendations.": "The response should demonstrate advanced analytical thinking by combining academic research (arXivPaper), current data (a search engine), market analysis (web browsing), consumer products (Amazon), and case studies (Wikipedia). The model should synthesize complex economic and environmental data into actionable insights.",
        },
    },
    {
        "name": "test_tool_usage_with_only_info_gathering_mcp_servers",
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": PRODUCTIVITY_INFO_MCPS,
        "queries": {
            # Basic tool access tests
            "List my GitHub repositories": "The response should use GitHub MCP to list the user's repositories with basic information like name and description.",
            "Show me the first 5 Linear issues": "The response should use Linear MCP to retrieve and display the first 5 issues with their titles and status.",
            "List the first 5 emails in my Gmail inbox": "The response should use Google Toolbox MCP to access Gmail and show the first 5 email subjects and senders.",
            "Show me recent messages from my Slack workspace": "The response should use Slack MCP to retrieve and display recent messages from available channels.",
            "List my Notion pages": "The response should use Notion MCP to show available pages and databases.",
            # Complex integration tests (existing)
            "can you list the files in my google drive? and if there exists a file identical to any in tanderson1299320's repository jSpreed let me know": "The response should use Google Drive MCP to list files and GitHub MCP to access the jSpreed repository, then compare files to identify any matches between the two locations.",
            "Based on the content of the files in the 'test_files' directory, can you identify which file is most likely to be a dataset of chemical elements and then find the discoverer of the first metal listed with a null molar heat?": "The response should use file system tools to list and read files in the 'test_files' directory, identify the JSON file with chemical elements, parse it, and find the requested information.",
            "There is a discussion in slack about adding features to the golang speedreader. Can you find that discussion and summarize the features that are being requested?": "The response should use Slack MCP to search for discussions about the 'golang speedreader' and summarize the requested features.",
            "There is an issue in our github repo about enriching the readme of the golang speedreader. Can you find it and tell me who is assigned to it?": "The response should use GitHub MCP to find the specified issue in the repository and identify the assignee.",
            "Can you compare all the files in my local test_files directory with the files in my Google Drive root folder and tell me which files exist in both locations, which are only local, and which are only in Google Drive? For any files that exist in both locations, check if their sizes or modification dates differ.": "The response should use file system tools to list and analyze files in the local 'test_files' directory, use Google Drive MCP to list files in the Drive root folder, then perform a comprehensive comparison of file names, sizes, and timestamps between the two locations.",
        },
    },
    {
        "name": "test_tool_usage_with_only_programming_mcp_servers",
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": {
            **PROGRAMMING_CODE_MCPS,
            **{"perplexity-search": BROWSING_SEARCH_MCPS["perplexity-search"]},
            **{"github": PRODUCTIVITY_INFO_MCPS["github"]},
            **{"slack": PRODUCTIVITY_INFO_MCPS["slack"]},
        },
        "queries": {
            # Basic tool access tests
            "List the files in the test_files directory": "The response should use filesystem MCP to list the contents of the test_files directory.",
            "Execute a simple Python script that prints 'Hello World'": "The response should use Python code execution MCP to run a basic print statement.",
            "Fetch the content from https://httpbin.org/json": "The response should use web fetch MCP to retrieve JSON data from the test endpoint.",
            "Access the jSpreed repository on GitHub and show me the README": "The response should use GitHub MCP to access the specified repository and display README content.",
            "Search Context7 for documentation about 'React hooks'": "The response should use Context7 MCP to search for React hooks documentation.",
            # Complex programming tasks (existing)
            "implement the changes to our go speedreader program at https://raw.githubusercontent.com/tanderson1299320/jSpreed/refs/heads/main/main.go specified in our linear": "The response should use GitHub MCP to access the Go code, Slack MCP or Linear-related tools to find specifications, and provide specific code implementation changes.",
            "re-write the go speedreader program at https://raw.githubusercontent.com/tanderson1299320/jSpreed/refs/heads/main/main.go in python and while doing so address the concerns about it in the slack": "The response should use GitHub MCP to get the Go code, Slack MCP to find concerns, and provide a complete Python rewrite that addresses the identified issues.",
            "list for me the newest possible versions of the packages in https://raw.githubusercontent.com/tanderson1299320/Gulp-AI/Osmosis-MCP-4B-demo/refs/heads/main/requirements.txt": "The response should use GitHub MCP to access the requirements.txt file and search tools to find the latest available versions of each listed package.",
            "In the golang speedreader program at https://raw.githubusercontent.com/tanderson1299320/jSpreed/refs/heads/main/main.go, add a function for calculating WPM based on a counter that starts when the program does": "The response should use GitHub MCP to get the Go code and provide a modified Go function that calculates both WPM and accuracy.",
            "The file 'test_files/d6059b3e-e1da-43b4-ac26-ecad2984909b.csv' contains data about LEGO parts. Can you write a python script that reads this file and creates a new CSV file containing only the parts that are black?": "The response should provide a Python script that uses a library like pandas or csv to read the specified CSV file, filter for rows where the color is 'Black', and write the result to a new CSV file.",
        },
    },
    {
        "name": "test_tool_usage_with_only_visual_mcp_servers",
        "mode": "experiment",
        "candidate_models": CANDIDATE_MODELS,
        "mcp_servers": {
            **VISUAL_CHART_MCPS,
            **{"browserbase": BROWSING_SEARCH_MCPS["browserbase"]},
            **{"google-toolbox": PRODUCTIVITY_INFO_MCPS["google-toolbox"]},
        },
        "queries": {
            # Basic tool access tests
            "Create a simple bar chart with sample data (apples: 10, oranges: 15, bananas: 8) with a chart mcp": "The response should contain a link to the chart",
            "Create a simple bar chart with sample data (apples: 10, oranges: 15, bananas: 8)": "The response should contain a link to the chart",
            "List the files in my Google Drive": "The response should use Google Toolbox MCP to access Google Drive and list available files.",
            "Navigate to https://crouton.net using the browser": "The response should use Browserbase MCP to navigate to the example website.",
            "Generate a simple UI component with ShadCN": "The response should use ShadCN UI MCP to create a basic UI component.",
            # Complex visual tasks (existing)
            "List files in my google drive then create a bar chart showing the number of types of files. Then look at the resulting image and tell me the trendline": "The response should use Google Drive MCP to list files, Chart MCP to create a bar chart of file types, and Visual Reasoning MCP to analyze the chart and describe file type distribution trends.",
            "Graph the dates of the emails sent to my gmail using the google-toolbox. Then look at the resulting image and tell me the trendline": "The response should contain trend information about the emails in my inbox.",
            "make a chart of computer color capabilites by year based on the data in https://en.wikipedia.org/wiki/List_of_color_palettes": "The response should use Browserbase MCP to access the Wikipedia page, extract color capability data by year, there should be a link to the chart in the response",
            "chart the distance relation between angkor wat, the roman colosseum, and tenochtitlan": "The response should use Google Maps MCP to get location coordinates, Chart MCP to create a distance visualization (possibly a triangle chart or distance matrix), and Visual Reasoning MCP to analyze the geographic relationships.",
            "Please take the data from 'test_files/7805912b-c8da-4134-9b54-b590f884352a.csv' and create a pie chart showing the precipitation distribution in the different boroughs of New York. Then, tell me which borough has the smallest share of precipitation.": "The response should use file system tools to read the CSV, Chart MCP to create a pie chart, and Visual Reasoning MCP to analyze the chart and identify the borough with the smallest precipitation share.",
        },
    },
]
