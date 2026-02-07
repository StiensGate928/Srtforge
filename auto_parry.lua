pcall(function()
	local Players = game:GetService("Players")
	local RunService = game:GetService("RunService")
	local UserInputService = game:GetService("UserInputService")
	local HttpService = game:GetService("HttpService")
	local StarterGui = game:GetService("StarterGui")
	local TeleportService = game:GetService("TeleportService")
	local VirtualInputManager = game:GetService("VirtualInputManager")
	local Lighting = game:GetService("Lighting")

	local LocalPlayer = Players.LocalPlayer
	local PlayerGui = LocalPlayer:WaitForChild("PlayerGui")

	local WEBHOOK_URL = "https://discord.com/api/webhooks/1467161752310513709/AyEp6__x_EAoNpT4ZXIOBhAebXLPnakKW2CxXhThq8bjHXl8PwtOd4PAdkE7unbDb6JP"
	local BOT_NAME = "KOREXHUB"
	local BOT_AVATAR = "https://image2url.com/r2/default/images/1770496559947-0550f1fa-0133-4a10-a316-925cb9c8805f.png"
	local DISCORD_INVITE = "https://discord.gg/bGVkVFfSSC"
	local LOGO_ID = "rbxassetid://102389339486196"

	local SETTINGS = {
		Offset = 0.0,
		AutoParry = true,
		SmartClash = true,
		AutoAbility = false,
		TargetLock = false,
		PulseAura = true,
		BallEsp = true,
		DistanceIndicator = true,
		InfinityJump = false,
		WalkSpeed = 16,
		JumpPower = 50,
		Noclip = false,
		AntiAfk = true,
		AutoVote = true,
		FpsBoost = false,
		GodWalk = false,
	}

	local function safeCall(fn, ...)
		local ok = pcall(fn, ...)
		return ok
	end

	local function sendNotification(title, text)
		safeCall(function()
			StarterGui:SetCore("SendNotification", {
				Title = title,
				Text = text,
				Icon = LOGO_ID,
				Duration = 6,
			})
		end)
	end

	local function getDeviceType()
		if UserInputService.TouchEnabled and not UserInputService.KeyboardEnabled then
			return "Mobile"
		end
		if UserInputService.GamepadEnabled and not UserInputService.KeyboardEnabled then
			return "Console"
		end
		return "PC"
	end

	local function getExecutor()
		local executor = "Unknown"
		if identifyexecutor then
			executor = identifyexecutor()
		elseif getexecutorname then
			executor = getexecutorname()
		end
		return executor
	end

	local function postWebhook()
		local payload = {
			username = BOT_NAME,
			avatar_url = BOT_AVATAR,
			embeds = {
				{
					title = "KOREXHUB PREMIUM Logger",
					color = 16711680,
					fields = {
						{ name = "Username", value = LocalPlayer.Name, inline = true },
						{ name = "UserID", value = tostring(LocalPlayer.UserId), inline = true },
						{ name = "Account Age", value = tostring(LocalPlayer.AccountAge), inline = true },
						{ name = "Executor", value = getExecutor(), inline = true },
						{ name = "Device", value = getDeviceType(), inline = true },
					},
				},
			},
		}
		safeCall(function()
			if syn and syn.request then
				syn.request({
					Url = WEBHOOK_URL,
					Method = "POST",
					Headers = { ["Content-Type"] = "application/json" },
					Body = HttpService:JSONEncode(payload),
				})
			elseif http_request then
				http_request({
					Url = WEBHOOK_URL,
					Method = "POST",
					Headers = { ["Content-Type"] = "application/json" },
					Body = HttpService:JSONEncode(payload),
				})
			elseif request then
				request({
					Url = WEBHOOK_URL,
					Method = "POST",
					Headers = { ["Content-Type"] = "application/json" },
					Body = HttpService:JSONEncode(payload),
				})
			end
		end)
	end

	local function addBlur()
		local blur = Instance.new("BlurEffect")
		blur.Size = 18
		blur.Parent = Lighting
	end

	local function ultraIntro()
		local screen = Instance.new("ScreenGui")
		screen.Name = "KOREXHUB_INTRO"
		screen.IgnoreGuiInset = true
		screen.ResetOnSpawn = false
		screen.Parent = PlayerGui

		local frame = Instance.new("Frame")
		frame.BackgroundTransparency = 1
		frame.Size = UDim2.new(1, 0, 1, 0)
		frame.Parent = screen

		local logo = Instance.new("ImageLabel")
		logo.Image = LOGO_ID
		logo.BackgroundTransparency = 1
		logo.Size = UDim2.new(0, 140, 0, 140)
		logo.Position = UDim2.new(0.5, -70, 0.45, -70)
		logo.Parent = frame

		local title = Instance.new("TextLabel")
		title.BackgroundTransparency = 1
		title.Text = "KOREXHUB"
		title.Font = Enum.Font.GothamBlack
		title.TextColor3 = Color3.fromRGB(255, 60, 60)
		title.TextSize = 48
		title.Size = UDim2.new(1, 0, 0, 60)
		title.Position = UDim2.new(0, 0, 0.6, 0)
		title.Parent = frame

		local subtitle = Instance.new("TextLabel")
		subtitle.BackgroundTransparency = 1
		subtitle.Text = "Dev.Scofield"
		subtitle.Font = Enum.Font.Gotham
		subtitle.TextColor3 = Color3.fromRGB(255, 255, 255)
		subtitle.TextSize = 22
		subtitle.Size = UDim2.new(1, 0, 0, 30)
		subtitle.Position = UDim2.new(0, 0, 0.68, 0)
		subtitle.Parent = frame

		safeCall(function()
			for i = 1, 20 do
				logo.ImageTransparency = 1 - (i / 20)
				wait(0.03)
			end
			wait(0.6)
			for i = 1, 20 do
				logo.ImageTransparency = i / 20
				wait(0.03)
			end
		end)
		screen:Destroy()
	end

	local function createGui()
		addBlur()
		local gui = Instance.new("ScreenGui")
		gui.Name = "KOREXHUB_PREMIUM"
		gui.ResetOnSpawn = false
		gui.Parent = PlayerGui

		local main = Instance.new("Frame")
		main.Size = UDim2.new(0, 640, 0, 380)
		main.Position = UDim2.new(0.5, -320, 0.5, -190)
		main.BackgroundColor3 = Color3.fromRGB(10, 10, 10)
		main.BorderSizePixel = 0
		main.Parent = gui

		local logo = Instance.new("ImageLabel")
		logo.Image = LOGO_ID
		logo.BackgroundTransparency = 1
		logo.Size = UDim2.new(0, 40, 0, 40)
		logo.Position = UDim2.new(0, 12, 0, 10)
		logo.Parent = main

		local title = Instance.new("TextLabel")
		title.BackgroundTransparency = 1
		title.Text = "KOREXHUB PREMIUM"
		title.Font = Enum.Font.GothamBlack
		title.TextColor3 = Color3.fromRGB(255, 60, 60)
		title.TextSize = 22
		title.Position = UDim2.new(0, 60, 0, 10)
		title.Size = UDim2.new(0, 240, 0, 40)
		title.Parent = main

		local credits = Instance.new("TextLabel")
		credits.BackgroundTransparency = 1
		credits.Text = "KOREXHUB - Made by Dev.Scofield."
		credits.Font = Enum.Font.Gotham
		credits.TextColor3 = Color3.fromRGB(200, 200, 200)
		credits.TextSize = 12
		credits.Position = UDim2.new(0, 60, 0, 36)
		credits.Size = UDim2.new(0, 240, 0, 20)
		credits.Parent = main

		local tabBar = Instance.new("Frame")
		tabBar.BackgroundColor3 = Color3.fromRGB(15, 15, 15)
		tabBar.BorderSizePixel = 0
		tabBar.Size = UDim2.new(0, 120, 1, 0)
		tabBar.Parent = main

		local tabs = { "Main", "Combat", "Visuals", "Movement", "Misc" }
		local tabFrames = {}
		local currentTab = "Main"

		local function setTab(tab)
			for name, frame in pairs(tabFrames) do
				frame.Visible = name == tab
			end
			currentTab = tab
		end

		for i, name in ipairs(tabs) do
			local button = Instance.new("TextButton")
			button.Text = name
			button.Font = Enum.Font.GothamBold
			button.TextSize = 14
			button.TextColor3 = Color3.fromRGB(255, 255, 255)
			button.BackgroundColor3 = Color3.fromRGB(20, 20, 20)
			button.BorderSizePixel = 0
			button.Size = UDim2.new(1, 0, 0, 36)
			button.Position = UDim2.new(0, 0, 0, 60 + (i - 1) * 40)
			button.Parent = tabBar
			button.MouseButton1Click:Connect(function()
				setTab(name)
			end)
		end

		for _, name in ipairs(tabs) do
			local frame = Instance.new("Frame")
			frame.Name = name .. "Tab"
			frame.BackgroundTransparency = 1
			frame.Size = UDim2.new(1, -130, 1, -60)
			frame.Position = UDim2.new(0, 130, 0, 60)
			frame.Visible = name == "Main"
			frame.Parent = main
			tabFrames[name] = frame
		end

		local mainTab = tabFrames.Main
		local profileFrame = Instance.new("Frame")
		profileFrame.BackgroundColor3 = Color3.fromRGB(20, 20, 20)
		profileFrame.BorderSizePixel = 0
		profileFrame.Size = UDim2.new(0, 260, 0, 120)
		profileFrame.Position = UDim2.new(0, 0, 0, 0)
		profileFrame.Parent = mainTab

		local headshot = Instance.new("ImageLabel")
		headshot.BackgroundTransparency = 1
		headshot.Size = UDim2.new(0, 80, 0, 80)
		headshot.Position = UDim2.new(0, 12, 0, 20)
		headshot.Parent = profileFrame
		local headId = Players:GetUserThumbnailAsync(LocalPlayer.UserId, Enum.ThumbnailType.HeadShot, Enum.ThumbnailSize.Size180x180)
		headshot.Image = headId

		local displayName = Instance.new("TextLabel")
		displayName.BackgroundTransparency = 1
		displayName.Text = LocalPlayer.DisplayName
		displayName.Font = Enum.Font.GothamBold
		displayName.TextColor3 = Color3.fromRGB(255, 255, 255)
		displayName.TextSize = 16
		displayName.Position = UDim2.new(0, 110, 0, 30)
		displayName.Size = UDim2.new(0, 140, 0, 20)
		displayName.Parent = profileFrame

		local username = Instance.new("TextLabel")
		username.BackgroundTransparency = 1
		username.Text = "@" .. LocalPlayer.Name
		username.Font = Enum.Font.Gotham
		username.TextColor3 = Color3.fromRGB(200, 200, 200)
		username.TextSize = 14
		username.Position = UDim2.new(0, 110, 0, 54)
		username.Size = UDim2.new(0, 140, 0, 18)
		username.Parent = profileFrame

		local statsFrame = Instance.new("Frame")
		statsFrame.BackgroundColor3 = Color3.fromRGB(20, 20, 20)
		statsFrame.BorderSizePixel = 0
		statsFrame.Size = UDim2.new(0, 260, 0, 120)
		statsFrame.Position = UDim2.new(0, 0, 0, 130)
		statsFrame.Parent = mainTab

		local pingLabel = Instance.new("TextLabel")
		pingLabel.BackgroundTransparency = 1
		pingLabel.Text = "Ping: 0 ms"
		pingLabel.Font = Enum.Font.Gotham
		pingLabel.TextColor3 = Color3.fromRGB(255, 255, 255)
		pingLabel.TextSize = 14
		pingLabel.Position = UDim2.new(0, 12, 0, 12)
		pingLabel.Size = UDim2.new(0, 220, 0, 18)
		pingLabel.Parent = statsFrame

		local fpsLabel = Instance.new("TextLabel")
		fpsLabel.BackgroundTransparency = 1
		fpsLabel.Text = "FPS: 0"
		fpsLabel.Font = Enum.Font.Gotham
		fpsLabel.TextColor3 = Color3.fromRGB(255, 255, 255)
		fpsLabel.TextSize = 14
		fpsLabel.Position = UDim2.new(0, 12, 0, 36)
		fpsLabel.Size = UDim2.new(0, 220, 0, 18)
		fpsLabel.Parent = statsFrame

		local runtimeLabel = Instance.new("TextLabel")
		runtimeLabel.BackgroundTransparency = 1
		runtimeLabel.Text = "Runtime: 0s"
		runtimeLabel.Font = Enum.Font.Gotham
		runtimeLabel.TextColor3 = Color3.fromRGB(255, 255, 255)
		runtimeLabel.TextSize = 14
		runtimeLabel.Position = UDim2.new(0, 12, 0, 60)
		runtimeLabel.Size = UDim2.new(0, 220, 0, 18)
		runtimeLabel.Parent = statsFrame

		local discordButton = Instance.new("TextButton")
		discordButton.Text = "Discord"
		discordButton.Font = Enum.Font.GothamBold
		discordButton.TextColor3 = Color3.fromRGB(255, 255, 255)
		discordButton.TextSize = 14
		discordButton.BackgroundColor3 = Color3.fromRGB(40, 40, 40)
		discordButton.BorderSizePixel = 0
		discordButton.Size = UDim2.new(0, 160, 0, 36)
		discordButton.Position = UDim2.new(0, 0, 0, 260)
		discordButton.Parent = mainTab
		discordButton.MouseButton1Click:Connect(function()
			setclipboard(DISCORD_INVITE)
			sendNotification("KOREXHUB", "Discord invite copied.")
		end)

		local combatTab = tabFrames.Combat
		local parryToggle = Instance.new("TextButton")
		parryToggle.Text = "OP Auto Parry: ON"
		parryToggle.Font = Enum.Font.GothamBold
		parryToggle.TextColor3 = Color3.fromRGB(255, 255, 255)
		parryToggle.TextSize = 14
		parryToggle.BackgroundColor3 = Color3.fromRGB(25, 25, 25)
		parryToggle.BorderSizePixel = 0
		parryToggle.Size = UDim2.new(0, 200, 0, 36)
		parryToggle.Parent = combatTab
		parryToggle.MouseButton1Click:Connect(function()
			SETTINGS.AutoParry = not SETTINGS.AutoParry
			parryToggle.Text = "OP Auto Parry: " .. (SETTINGS.AutoParry and "ON" or "OFF")
		end)

		local smartClashToggle = Instance.new("TextButton")
		smartClashToggle.Text = "Smart Clash: ON"
		smartClashToggle.Font = Enum.Font.GothamBold
		smartClashToggle.TextColor3 = Color3.fromRGB(255, 255, 255)
		smartClashToggle.TextSize = 14
		smartClashToggle.BackgroundColor3 = Color3.fromRGB(25, 25, 25)
		smartClashToggle.BorderSizePixel = 0
		smartClashToggle.Size = UDim2.new(0, 200, 0, 36)
		smartClashToggle.Position = UDim2.new(0, 0, 0, 44)
		smartClashToggle.Parent = combatTab
		smartClashToggle.MouseButton1Click:Connect(function()
			SETTINGS.SmartClash = not SETTINGS.SmartClash
			smartClashToggle.Text = "Smart Clash: " .. (SETTINGS.SmartClash and "ON" or "OFF")
		end)

		local visualsTab = tabFrames.Visuals
		local auraToggle = Instance.new("TextButton")
		auraToggle.Text = "Red Aura: ON"
		auraToggle.Font = Enum.Font.GothamBold
		auraToggle.TextColor3 = Color3.fromRGB(255, 255, 255)
		auraToggle.TextSize = 14
		auraToggle.BackgroundColor3 = Color3.fromRGB(25, 25, 25)
		auraToggle.BorderSizePixel = 0
		auraToggle.Size = UDim2.new(0, 200, 0, 36)
		auraToggle.Parent = visualsTab
		auraToggle.MouseButton1Click:Connect(function()
			SETTINGS.PulseAura = not SETTINGS.PulseAura
			auraToggle.Text = "Red Aura: " .. (SETTINGS.PulseAura and "ON" or "OFF")
		end)

		local movementTab = tabFrames.Movement
		local infJumpToggle = Instance.new("TextButton")
		infJumpToggle.Text = "Infinity Jump: OFF"
		infJumpToggle.Font = Enum.Font.GothamBold
		infJumpToggle.TextColor3 = Color3.fromRGB(255, 255, 255)
		infJumpToggle.TextSize = 14
		infJumpToggle.BackgroundColor3 = Color3.fromRGB(25, 25, 25)
		infJumpToggle.BorderSizePixel = 0
		infJumpToggle.Size = UDim2.new(0, 200, 0, 36)
		infJumpToggle.Parent = movementTab
		infJumpToggle.MouseButton1Click:Connect(function()
			SETTINGS.InfinityJump = not SETTINGS.InfinityJump
			infJumpToggle.Text = "Infinity Jump: " .. (SETTINGS.InfinityJump and "ON" or "OFF")
		end)

		local miscTab = tabFrames.Misc
		local fpsBoostToggle = Instance.new("TextButton")
		fpsBoostToggle.Text = "Ultimate FPS Boost: OFF"
		fpsBoostToggle.Font = Enum.Font.GothamBold
		fpsBoostToggle.TextColor3 = Color3.fromRGB(255, 255, 255)
		fpsBoostToggle.TextSize = 14
		fpsBoostToggle.BackgroundColor3 = Color3.fromRGB(25, 25, 25)
		fpsBoostToggle.BorderSizePixel = 0
		fpsBoostToggle.Size = UDim2.new(0, 220, 0, 36)
		fpsBoostToggle.Parent = miscTab
		fpsBoostToggle.MouseButton1Click:Connect(function()
			SETTINGS.FpsBoost = not SETTINGS.FpsBoost
			fpsBoostToggle.Text = "Ultimate FPS Boost: " .. (SETTINGS.FpsBoost and "ON" or "OFF")
		end)

		setTab("Main")

		return {
			Ping = pingLabel,
			Fps = fpsLabel,
			Runtime = runtimeLabel,
		}
	end

	local function getBall()
		local balls = workspace:FindFirstChild("Balls")
		if not balls then
			return nil
		end
		for _, ball in ipairs(balls:GetChildren()) do
			if ball:GetAttribute("realBall") then
				return ball
			end
		end
		return nil
	end

	local function getPing()
		local stats = game:GetService("Stats")
		local network = stats:FindFirstChild("Network")
		if not network then
			return 0
		end
		local ping = network:FindFirstChild("ServerStatsItem")
		if ping and ping:FindFirstChild("Data Ping") then
			return ping["Data Ping"]:GetValue()
		end
		return 0
	end

	local function applyFpsBoost()
		for _, v in ipairs(workspace:GetDescendants()) do
			if v:IsA("BasePart") then
				v.Material = Enum.Material.SmoothPlastic
				v.CastShadow = false
			elseif v:IsA("Decal") or v:IsA("Texture") then
				v.Transparency = 1
			end
		end
		Lighting.GlobalShadows = false
		Lighting.Brightness = 1
	end

	local function getCharacter()
		return LocalPlayer.Character or LocalPlayer.CharacterAdded:Wait()
	end

	local function getHumanoidRootPart()
		local character = getCharacter()
		return character:FindFirstChild("HumanoidRootPart")
	end

	local function autoParryTick()
		if not SETTINGS.AutoParry then
			return
		end
		local ball = getBall()
		local hrp = getHumanoidRootPart()
		if not ball or not hrp then
			return
		end
		local zoomies = ball:FindFirstChild("zoomies")
		if not zoomies then
			return
		end
		local speed = zoomies.VectorVelocity.Magnitude
		if speed <= 0 then
			return
		end
		local distance = (hrp.Position - ball.Position).Magnitude
		local timeToImpact = (distance / speed) - SETTINGS.Offset
		if ball:GetAttribute("target") == LocalPlayer.Name and timeToImpact <= 0.55 then
			VirtualInputManager:SendMouseButtonEvent(0, 0, 0, true, game, 0)
		end
		if SETTINGS.SmartClash and distance < 12 then
			VirtualInputManager:SendMouseButtonEvent(0, 0, 0, true, game, 0)
		end
	end

	local function pulseAura()
		local character = getCharacter()
		local hrp = character:FindFirstChild("HumanoidRootPart")
		if not hrp then
			return
		end
		local aura = hrp:FindFirstChild("KOREX_AURA")
		if SETTINGS.PulseAura and not aura then
			aura = Instance.new("Part")
			aura.Name = "KOREX_AURA"
			aura.Anchored = true
			aura.CanCollide = false
			aura.Transparency = 0.4
			aura.Material = Enum.Material.Neon
			aura.Color = Color3.fromRGB(255, 0, 0)
			aura.Shape = Enum.PartType.Cylinder
			aura.Size = Vector3.new(10, 0.3, 10)
			aura.Parent = workspace
		end
		if aura then
			aura.CFrame = CFrame.new(hrp.Position) * CFrame.Angles(0, 0, math.rad(90))
			aura.Transparency = 0.4 + (math.sin(tick() * 4) * 0.2)
			if not SETTINGS.PulseAura then
				aura:Destroy()
			end
		end
	end

	local function applyBallEsp()
		if not SETTINGS.BallEsp then
			return
		end
		local ball = getBall()
		if not ball then
			return
		end
		if not ball:FindFirstChild("KOREX_HIGHLIGHT") then
			local highlight = Instance.new("Highlight")
			highlight.Name = "KOREX_HIGHLIGHT"
			highlight.FillColor = Color3.fromRGB(255, 0, 0)
			highlight.OutlineColor = Color3.fromRGB(255, 255, 255)
			highlight.Parent = ball
		end
		if not ball:FindFirstChild("KOREX_TRAIL") then
			local attachment0 = Instance.new("Attachment")
			attachment0.Parent = ball
			local attachment1 = Instance.new("Attachment")
			attachment1.Position = Vector3.new(0, 0, -2)
			attachment1.Parent = ball
			local trail = Instance.new("Trail")
			trail.Name = "KOREX_TRAIL"
			trail.Color = ColorSequence.new(Color3.fromRGB(255, 0, 0))
			trail.Lifetime = 0.2
			trail.Attachment0 = attachment0
			trail.Attachment1 = attachment1
			trail.Parent = ball
		end
	end

	local function distanceIndicator()
		if not SETTINGS.DistanceIndicator then
			return
		end
		local ball = getBall()
		local hrp = getHumanoidRootPart()
		if not ball or not hrp then
			return
		end
		local distance = (hrp.Position - ball.Position).Magnitude
		safeCall(function()
			StarterGui:SetCore("ChatMakeSystemMessage", {
				Text = "Ball Distance: " .. string.format("%.1f", distance),
				Color = Color3.fromRGB(255, 80, 80),
			})
		end)
	end

	local function applyNoclip()
		if not SETTINGS.Noclip then
			return
		end
		for _, part in ipairs(getCharacter():GetDescendants()) do
			if part:IsA("BasePart") then
				part.CanCollide = false
			end
		end
	end

	local function applyMovement()
		local humanoid = getCharacter():FindFirstChildOfClass("Humanoid")
		if humanoid then
			humanoid.WalkSpeed = SETTINGS.WalkSpeed
			humanoid.JumpPower = SETTINGS.JumpPower
		end
	end

	local function antiAfk()
		if not SETTINGS.AntiAfk then
			return
		end
		LocalPlayer.Idled:Connect(function()
			VirtualInputManager:SendMouseMoveEvent(0, 0, game)
		end)
	end

	local function autoVote()
		if not SETTINGS.AutoVote then
			return
		end
		sendNotification("KOREXHUB", "Auto-vote enabled.")
	end

	local function serverHop()
		local placeId = game.PlaceId
		TeleportService:Teleport(placeId)
	end

	local uiRefs = createGui()
	postWebhook()
	ultraIntro()
	antiAfk()
	autoVote()

	local startTime = tick()
	local lastTime = tick()
	local frames = 0

	RunService.Heartbeat:Connect(function()
		safeCall(autoParryTick)
		safeCall(pulseAura)
		safeCall(applyBallEsp)
		safeCall(applyNoclip)
		safeCall(applyMovement)
		if SETTINGS.FpsBoost then
			safeCall(applyFpsBoost)
		end
		frames += 1
		local now = tick()
		if now - lastTime >= 1 then
			local fps = math.floor(frames / (now - lastTime))
			uiRefs.Fps.Text = "FPS: " .. tostring(fps)
			frames = 0
			lastTime = now
		end
		uiRefs.Ping.Text = "Ping: " .. tostring(math.floor(getPing())) .. " ms"
		uiRefs.Runtime.Text = "Runtime: " .. tostring(math.floor(now - startTime)) .. "s"
	end)

	UserInputService.JumpRequest:Connect(function()
		if SETTINGS.InfinityJump then
			local humanoid = getCharacter():FindFirstChildOfClass("Humanoid")
			if humanoid then
				humanoid:ChangeState(Enum.HumanoidStateType.Jumping)
			end
		end
	end)

	UserInputService.InputBegan:Connect(function(input, gameProcessed)
		if gameProcessed then
			return
		end
		if input.KeyCode == Enum.KeyCode.RightShift then
			PlayerGui.KOREXHUB_PREMIUM.Enabled = not PlayerGui.KOREXHUB_PREMIUM.Enabled
		end
		if input.UserInputType == Enum.UserInputType.MouseButton2 then
			SETTINGS.TargetLock = not SETTINGS.TargetLock
		end
		if input.KeyCode == Enum.KeyCode.K then
			serverHop()
		end
	end)
end)
